import math
import torch.optim.lr_scheduler as lr_sched


def get_scheduler(optimizer, cfg):
    name = cfg.get('training', cfg).get('scheduler', 'cosine')
    epochs = cfg.get('training', cfg).get('epochs', 300)
    warmup = cfg.get('training', cfg).get('warmup_epochs', 10)

    if name == 'cosine':
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_sched.LambdaLR(optimizer, lr_lambda)

    if name == 'step':
        return lr_sched.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)

    if name == 'poly':
        def poly_lambda(epoch):
            return (1.0 - epoch / epochs) ** 0.9
        return lr_sched.LambdaLR(optimizer, poly_lambda)

    return lr_sched.ConstantLR(optimizer, factor=1.0)
