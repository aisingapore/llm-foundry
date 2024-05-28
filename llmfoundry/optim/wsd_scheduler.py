
import logging

import textwrap
import warnings
from typing import TYPE_CHECKING, Union

from composer.optim import (ComposerScheduler,
                            LinearScheduler, 
                            _convert_time, 
                            _raise_if_max_duration_exceeds_t_max,
                            _raise_if_warmup_and_max_incompatible)
from composer.core import State, Time

if TYPE_CHECKING:
    from typing import Protocol  # type: ignore
else:
    # subclasses of Protocol cannot be instantiated in Python 3.8
    Protocol = object

log = logging.getLogger(__name__)

class WSDScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly, with an initial warmup period and a final cooldown period.


    Args:
        t_warmup (str | Time): Warmup time.
        t_decay (str | Time): Decay time.
        alpha_i (float): Constant learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_decay: Union[str, Time],
        alpha_i: float = 1.0,
        alpha_f: float = 0.0,
        t_max: Union[str, Time] = '1dur',
        scale_warmup: bool = False,
    ):
        self.t_warmup = t_warmup
        self.t_decap = t_decay
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=alpha_i, t_max=t_warmup)
        self.decay_scheduler = LinearScheduler(alpha_i=alpha_i, alpha_f=alpha_f, t_max=t_decay)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        t_warmup = _convert_time(self.t_warmup, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        t_decay = _convert_time(self.t_decay, state, ssr=ssr)
        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent(
                    """\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter.""",
                ),
            )

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        
        if state.timestamp > t_decay:
            
            return self.decay_scheduler(state)

        return self.alpha_i
