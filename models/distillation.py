import torch
import torch.nn as nn
import torch.nn.functional as F
from .meamt_net import MEaMtNet


class FeatureProjector(nn.Module):
    """Projects features to a shared embedding space for contrastive learning."""

    def __init__(self, in_ch, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)


class Distiller(nn.Module):
    """
    Teacher-student distillation module.

    Teacher: trained on contrast-enhanced MRI (CE-MRI), frozen during student training.
    Student: trained on non-contrast MRI (T2FS + DWI) with supervised + distillation losses.
    Contrastive pairs are formed between teacher and student feature projections.
    """

    def __init__(self, teacher: MEaMtNet, student: MEaMtNet, proj_dim=128):
        super().__init__()
        self.teacher = teacher
        self.student = student

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Determine bottleneck channel from student
        base = student.stem[0].out_channels
        bn_ch = base * 16

        self.teacher_projector = FeatureProjector(bn_ch, proj_dim)
        self.student_projector = FeatureProjector(bn_ch, proj_dim)

    def forward(self, student_input, teacher_input=None):
        student_out = self.student(student_input)
        s_feat = student_out['features']
        s_proj = self.student_projector(s_feat)

        t_proj = None
        t_out = None
        if teacher_input is not None and self.training:
            with torch.no_grad():
                t_out = self.teacher(teacher_input)
            t_feat = t_out['features']
            t_proj = self.teacher_projector(t_feat)

        student_out['student_proj'] = s_proj
        student_out['teacher_proj'] = t_proj
        student_out['teacher_out'] = t_out
        return student_out
