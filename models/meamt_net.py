import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoders import SegDecoder, QuantDecoder
from .evidence_head import EvidenceHead

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + x)


class EdgeGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        att = self.gate(torch.cat([x, edge], dim=1))
        return x * att + edge * (1.0 - att)


class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = nn.MultiheadAttention(dim, num_heads=max(1, dim // 64), batch_first=True)
        self.use_mamba = HAS_MAMBA

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)          # [B, H*W, C]
        normed = self.norm(seq)
        if self.use_mamba:
            out = self.ssm(normed)
        else:
            out, _ = self.ssm(normed, normed, normed)
        return (out + seq).transpose(1, 2).reshape(B, C, H, W)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, use_edge=True):
        super().__init__()
        self.downsample = ConvBnRelu(in_ch, out_ch, stride=2)
        self.res = ResBlock(out_ch)
        self.edge_att = EdgeGuidedAttention(out_ch) if use_edge else nn.Identity()

    def forward(self, x):
        x = self.edge_att(self.res(self.downsample(x)))
        return x


class MEaMtNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg.get('in_channels', 2)
        base = cfg.get('base_channels', 64)
        num_cls = cfg.get('num_classes', 2)
        quant_dim = cfg.get('quant_dim', 3)
        use_edge = cfg.get('use_edge_attention', True)
        use_mamba = cfg.get('use_mamba', True)
        use_evi = cfg.get('use_evidence', True)

        self.stem = ConvBnRelu(in_ch, base, kernel=7, stride=1, padding=3)

        self.enc1 = EncoderStage(base, base * 2, use_edge)
        self.enc2 = EncoderStage(base * 2, base * 4, use_edge)
        self.enc3 = EncoderStage(base * 4, base * 8, use_edge)

        self.bottleneck = nn.Sequential(
            ConvBnRelu(base * 8, base * 16, stride=2),
            ResBlock(base * 16),
            MambaBlock(base * 16) if use_mamba else nn.Identity(),
            ResBlock(base * 16),
        )

        self.seg_decoder = SegDecoder(
            enc_channels=[base, base * 2, base * 4, base * 8],
            bottleneck_ch=base * 16,
            num_classes=num_cls,
        )
        self.quant_decoder = QuantDecoder(base * 16, quant_dim)

        self.evidence_head = EvidenceHead(num_cls, quant_dim) if use_evi else None

    def forward(self, x):
        s0 = self.stem(x)                 # [B, 64, H, W]
        s1 = self.enc1(s0)                # [B, 128, H/2, W/2]
        s2 = self.enc2(s1)                # [B, 256, H/4, W/4]
        s3 = self.enc3(s2)                # [B, 512, H/8, W/8]
        bot = self.bottleneck(s3)         # [B, 1024, H/16, W/16]

        seg_logits = self.seg_decoder(bot, [s0, s1, s2, s3])
        quant_pred = self.quant_decoder(bot)

        out = {'seg': seg_logits, 'quant': quant_pred, 'features': bot}

        if self.evidence_head is not None:
            evi_out = self.evidence_head(seg_logits, quant_pred)
            out.update(evi_out)

        return out
