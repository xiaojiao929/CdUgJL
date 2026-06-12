import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from configs.default import load_config
from data.dataset import MedicalImageDataset
from data.transforms import get_transforms
from models.meamt_net import MEaMtNet
from models.distillation import Distiller
from losses.segmentation_loss import DiceCELoss
from losses.distillation_loss import DistillationLoss
from losses.evidence_loss import EvidenceLoss
from utils.logger import Logger
from utils.scheduler import get_scheduler
from eval.inference import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = cfg['logging']['log_dir']
    save_dir = cfg['logging']['save_path']
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(log_dir)

    t_cfg = cfg['training']
    d_cfg = cfg['dataset']
    m_cfg = cfg['model']
    l_cfg = cfg['loss']

    input_size = tuple(d_cfg['input_size'])
    train_tfms, val_tfms = get_transforms(input_size)

    train_ds = MedicalImageDataset(
        d_cfg['root'], mode='train',
        transforms=train_tfms,
        label_ratio=d_cfg.get('label_ratio', 1.0),
    )
    val_ds = MedicalImageDataset(d_cfg['root'], mode='val', transforms=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=t_cfg['batch_size'],
                              shuffle=True, num_workers=t_cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=t_cfg['num_workers'])

    teacher = MEaMtNet(m_cfg).to(device)
    student = MEaMtNet(m_cfg).to(device)
    distiller = Distiller(teacher, student).to(device)

    if args.resume:
        student.load_state_dict(torch.load(args.resume, map_location=device))
        logger.log(f"Resumed from {args.resume}")

    loss_seg = DiceCELoss()
    loss_distill = DistillationLoss(
        temperature=l_cfg.get('contrastive_temperature', 0.07),
        feat_weight=l_cfg.get('distill_weight', 0.5),
        contra_weight=l_cfg.get('contrastive_weight', 0.4),
    )
    loss_evi = EvidenceLoss(beta=l_cfg.get('beta', 0.6))

    optimizer = optim.AdamW(distiller.parameters(),
                            lr=t_cfg['learning_rate'],
                            weight_decay=t_cfg['weight_decay'])
    scheduler = get_scheduler(optimizer, cfg)

    best_dice = 0.0
    grad_clip = t_cfg.get('gradient_clip', 5.0)

    for epoch in range(t_cfg['epochs']):
        distiller.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['seg'].to(device)
            quant_gt = batch['quant'].to(device)
            labeled = batch['labeled']

            ce_images = batch['ce_image'].to(device)
            has_ce = batch['has_ce']

            optimizer.zero_grad()
            # Pass CE-MRI to teacher when available
            teacher_input = ce_images if has_ce.any() else None
            out = distiller(images, teacher_input=teacher_input)

            # Supervised losses only on labeled samples
            labeled_mask = labeled.bool().to(device)
            if labeled_mask.any():
                seg_loss = loss_seg(out['seg'][labeled_mask],
                                    masks[labeled_mask].clamp(min=0))
                # Only slice tensor leaves; skip nested dicts (evidence/uncertainty)
                evi_inputs = {
                    k: v[labeled_mask]
                    if (isinstance(v, torch.Tensor) and v.shape[0] == images.shape[0])
                    else v
                    for k, v in out.items()
                    if not isinstance(v, dict)
                }
                evi_loss = loss_evi(
                    evi_inputs,
                    masks[labeled_mask].clamp(min=0),
                    quant_gt[labeled_mask],
                )
            else:
                seg_loss = torch.tensor(0.0, device=device)
                evi_loss = torch.tensor(0.0, device=device)

            distill_loss = loss_distill(out)

            total = (l_cfg['seg_weight'] * seg_loss
                     + l_cfg['distill_weight'] * distill_loss
                     + l_cfg['evidence_weight'] * evi_loss)
            total.backward()
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += total.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.log(f"[Epoch {epoch+1}/{t_cfg['epochs']}] Loss: {avg_loss:.4f}")
        logger.scalar('train/loss', avg_loss, epoch)

        if (epoch + 1) % cfg['logging']['eval_interval'] == 0:
            dice = evaluate_model(student, val_loader, device, d_cfg['num_classes'])
            logger.log(f"[Epoch {epoch+1}] Val Dice: {dice:.4f}")
            logger.scalar('val/dice', dice, epoch)

            if dice > best_dice:
                best_dice = dice
                torch.save(student.state_dict(),
                           os.path.join(save_dir, 'meamtnet_best.pth'))
                logger.log(f"  => Saved best model (Dice={best_dice:.4f})")

        if (epoch + 1) % cfg['logging']['save_interval'] == 0:
            torch.save(student.state_dict(),
                       os.path.join(save_dir, f'meamtnet_epoch{epoch+1}.pth'))

    logger.log(f"Training complete. Best Val Dice: {best_dice:.4f}")
    logger.close()


if __name__ == '__main__':
    main()
