import torch
from torch import optim


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['_step'] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        if '_step' in state_dict:
            self._step = state_dict['_step']
            del state_dict['_step']
        self.optimizer.load_state_dict(state_dict)


def get_std_opt(model, optim_func='adam', factor=1, warmup=2000):
    optim_func = dict(Adam=torch.optim.Adam,
                      AdamW=torch.optim.AdamW)[optim_func]
    return NoamOpt(model.d_model, factor, warmup,
                   optim_func(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def build_noamopt_optimizer(args, model):
    ve_optimizer = getattr(torch.optim, args.optim)(
        model.visual_extractor.parameters(),
        lr=0,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ed_optimizer = get_std_opt(model.encoder_decoder, optim_func=args.optim, factor=args.noamopt_factor,
                               warmup=args.noamopt_warmup)
    return ve_optimizer, ed_optimizer


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor,
                                                              patience=patience, verbose=verbose, threshold=threshold,
                                                              threshold_mode=threshold_mode, cooldown=cooldown,
                                                              min_lr=min_lr, eps=eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def build_plateau_optimizer(args, model):
    ve_optimizer = getattr(torch.optim, args.optim)(
        model.visual_extractor.parameters(),
        lr=args.lr_ve,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ve_optimizer = ReduceLROnPlateau(ve_optimizer,
                                     factor=args.reduce_on_plateau_factor,
                                     patience=args.reduce_on_plateau_patience)
    ed_optimizer = getattr(torch.optim, args.optim)(
        model.encoder_decoder.parameters(),
        lr=args.lr_ed,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    ed_optimizer = ReduceLROnPlateau(ed_optimizer,
                                     factor=args.reduce_on_plateau_factor,
                                     patience=args.reduce_on_plateau_patience)

    return ve_optimizer, ed_optimizer
