import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import load_config
from data.dataset import MedicalImageDataset
from data.transforms import get_transforms
from models.meamt_net import MEaMtNet
from utils.metrics import compute_all_metrics
from eval.inference import run_inference
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--tta', action='store_true', help='Test-time augmentation')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    d_cfg = cfg['dataset']
    t_cfg = cfg['testing']
    m_cfg = cfg['model']

    output_dir = t_cfg.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger('MEaMtNet-Test', output=output_dir, filename='test_log.txt')
    logger.log(f"Config: {args.config}  Checkpoint: {args.checkpoint}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_tfms = get_transforms(tuple(d_cfg['input_size']))
    test_ds = MedicalImageDataset(d_cfg['root'], mode='test', transforms=val_tfms)
    test_loader = DataLoader(test_ds, batch_size=t_cfg.get('batch_size', 1),
                             shuffle=False, num_workers=cfg['training']['num_workers'])

    model = MEaMtNet(m_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    logger.log(f"Loaded checkpoint: {args.checkpoint}")

    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            gt_seg = batch['seg'].to(device)
            gt_quant = batch['quant'].to(device)

            pred_seg, pred_quant, _ = run_inference(model, images, tta=args.tta)

            if (gt_seg >= 0).any():
                metrics = compute_all_metrics(pred_seg, gt_seg, pred_quant, gt_quant,
                                              d_cfg['num_classes'])
                all_metrics.append(metrics)

            if args.save_results:
                for i, pid in enumerate(batch['id']):
                    torch.save(pred_seg[i].cpu(), os.path.join(output_dir, f'{pid}_seg.pt'))
                    torch.save(pred_quant[i].cpu(), os.path.join(output_dir, f'{pid}_quant.pt'))

    if all_metrics:
        mean = {k: sum(m[k] for m in all_metrics if not (m[k] != m[k])) / len(all_metrics)
                for k in all_metrics[0]}
        logger.log("Results:")
        for k, v in mean.items():
            logger.log(f"  {k}: {v:.4f}")
    else:
        logger.log("No labeled test samples found — predictions saved only.")


if __name__ == '__main__':
    main()
