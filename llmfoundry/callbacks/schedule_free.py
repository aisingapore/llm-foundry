from warnings import warn

from composer.core import Callback, State
from composer.loggers import Logger
from llmfoundry.optim import ScheduleFree

class ScheduleFreeCallback(Callback):
    """Callback specifically for Schedule Free training.
    """

    def after_load(self, state: State, logger: Logger) -> None:
        # check if there is any schedule free optimizer
        if isinstance(state.optimizer, list):
            check_optimizer = any(isinstance(optimizer, ScheduleFree) for 
                                  optimizer in state.optimizer)

        elif isinstance(state.optimizer, ScheduleFree):
            
            check_optimizer = True

        if not check_optimizer:
            warn("Optimizer is not Schedule Free. This callback will not work.")
    def fit_start(self, state: State, logger: Logger) -> None:
        # check if optimizer is AdamWScheduleFree

        if isinstance(state.optimizer, ScheduleFree):
            state.optimizer.train()


    def eval_start(self, state: State, logger: Logger) -> None:
        
        if isinstance(state.optimizer, ScheduleFree):
            state.optimizer.eval()