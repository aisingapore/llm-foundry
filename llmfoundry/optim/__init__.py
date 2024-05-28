# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.optim import (ConstantWithWarmupScheduler,
                            CosineAnnealingWithWarmupScheduler, DecoupledAdamW,
                            LinearWithWarmupScheduler)

from llmfoundry.optim.adaptive_lion import DecoupledAdaLRLion, DecoupledClipLion
from llmfoundry.optim.lion import DecoupledLionW
from llmfoundry.optim.wsd_scheduler import WSDScheduler
from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler
from llmfoundry.optim.adamw_schedulefree import AdamWScheduleFree, ScheduleFree
from llmfoundry.registry import optimizers, schedulers

optimizers.register('adalr_lion', func=DecoupledAdaLRLion)
optimizers.register('clip_lion', func=DecoupledClipLion)
optimizers.register('decoupled_lionw', func=DecoupledLionW)
optimizers.register('decoupled_adamw', func=DecoupledAdamW)
optimizers.register('adamw_schedulefree', func=AdamWScheduleFree)

schedulers.register('constant_with_warmup', func=ConstantWithWarmupScheduler)
schedulers.register('cosine_with_warmup',
                    func=CosineAnnealingWithWarmupScheduler)
schedulers.register('linear_decay_with_warmup', func=LinearWithWarmupScheduler)
schedulers.register('inv_sqrt_with_warmup',
                    func=InverseSquareRootWithWarmupScheduler)
schedulers.register('wsd_scheduler', func=WSDScheduler)

__all__ = [
    'ScheduleFree',
    'DecoupledLionW',
    'DecoupledClipLion',
    'DecoupledAdaLRLion',
    'DecoupledAdamW',
    'AdamWScheduleFree',
]
