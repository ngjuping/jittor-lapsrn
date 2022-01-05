class LRScheduler:
    def __init__(self, optimizer, base_lr):
        self.optimizer = optimizer

        self.basic_lr = base_lr
        self.lr_decay = 0.6
        self.decay_step = 15000

    def step(self, step):
        lr_decay = self.lr_decay ** int(step / self.decay_step)
        lr_decay = max(lr_decay, 2e-5)
        self.optimizer.lr = lr_decay * self.basic_lr