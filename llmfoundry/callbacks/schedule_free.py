from warnings import warn

from composer.core import Callback, State
from composer.loggers import Logger
from llmfoundry.optim import ScheduleFree

class ScheduleFreeCallback(Callback):
    """Callback specifically for Schedule Free training.
    """

    def after_load(self, state: State, logger: Logger) -> None:
        # check if there is any schedule free optimizer

        del logger  # unused
        if isinstance(state.optimizer, list):
            check_optimizer = any(isinstance(optimizer, ScheduleFree) for 
                                  optimizer in state.optimizer)

        elif isinstance(state.optimizer, ScheduleFree):
            
            check_optimizer = True

        if not check_optimizer:
            warn("Optimizer is not Schedule Free. This callback will not work.")
    def fit_start(self, state: State, logger: Logger) -> None:
        # check if optimizer is AdamWScheduleFree

        del logger  # unused
        if isinstance(state.optimizer, ScheduleFree):
            state.optimizer.train()

        elif isinstance(state.optimizer, list):
            for optimizer in state.optimizer:
                if isinstance(optimizer, ScheduleFree):
                    optimizer.train()


    def eval_start(self, state: State, logger: Logger) -> None:

        del logger  # unused

        if isinstance(state.optimizer, ScheduleFree):
            state.optimizer.eval()

        elif isinstance(state.optimizer, list):
            for optimizer in state.optimizer:
                if isinstance(optimizer, ScheduleFree):
                    optimizer.eval()


    def fit_end(self, state: State, logger: Logger) -> None:
        """
        Copied from eval_start due to the same implementation
        """
        self.eval_start(state, logger)