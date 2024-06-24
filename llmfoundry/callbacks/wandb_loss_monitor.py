import logging
import os
from collections import deque
from datetime import datetime
from typing import Optional, Deque, Tuple

import numpy as np
import numpy as np
import torch
from composer.core import State
from composer.loggers import Logger
from composer.utils import dist
from scipy.stats import linregress

from llmfoundry.interfaces import CallbackWithConfig
import wandb


log = logging.getLogger(__name__)


class WAndBLossMonitor(CallbackWithConfig):

    def __init__(
        self,
        window_size: int = 100,
        frequency_threshold: float = 0.6,
        magnitude_threshold: float = 0.05,
        slope_threshold: float = 0.1,
        alert_frequency: int = 300,
        report_ok: bool = False,
    ) -> None:
        self.alert_frequency = alert_frequency
        self.report_ok = report_ok

        # Ensure we can check as soon as we sample enough data
        self.last_alert = -self.alert_frequency
        self.checker = LossDivergenceChecker(
            window_size, frequency_threshold, magnitude_threshold, slope_threshold
        )
    def batch_end(self, state: State, logger: Logger) -> None:
        # Only need to run on master process
        if dist.get_global_rank() != 0:
            return

        if not isinstance(state.loss, torch.Tensor):
            raise NotImplementedError("Multiple losses not supported.")
        step = state.timestamp.batch.value
        loss = state.loss.item()
        now = state.timestamp.total_wct.seconds

        div_start = 500
        div_dur = 200
        if step > div_start:
            loss *= ((step - div_start) % div_dur) / div_dur * 5.0

        self.checker.sample(step, loss)

        if self._check(now):
            message, alert = self.checker.check(loss)
            if alert or (self.report_ok and message is not None):
                self._alert(message, state.run_name)
                self.last_alert = now

            return alert

    def _alert(self, message:str, run_name:str) -> None:
        prefix = f"[{datetime.now()}][{run_name}][node_rank={dist.get_node_rank()}]"
        node_name = os.environ.get("NODENAME", None)
        if node_name is not None:
            prefix += f"[node={node_name}]"
        
        wandb.alert(title = prefix, 
                    text = message, 
                    level = wandb.AlertLevel.WARN,
                    wait_duration = 300)
        
    def _check(self, wallclock_time: int) -> bool:
        return wallclock_time - self.last_alert >= self.alert_frequency

class LossDivergenceChecker:
    def __init__(
        self,
        window_size: int,
        frequency_threshold: float,
        magnitude_threshold: float,
        slope_threshold: float,
    ) -> None:
        self.window_size = window_size
        self.frequency_threshold = frequency_threshold
        self.magnitude_threshold = magnitude_threshold
        self.slope_threshold = slope_threshold

        self.steps: Deque[int] = deque(maxlen=self.window_size)
        self.losses: Deque[float] = deque(maxlen=self.window_size)

    def check(self, loss: float) -> Tuple[Optional[str], bool]:
        # Skip if we have not sampled enough data points
        if len(self.losses) != self.losses.maxlen:
            return None, False

        message = "No divergence"
        min_loss = min(self.losses)
        loss_range = max(self.losses) - min_loss

        orig_loss = self.losses[0]
        # Skip if loss is converging at the end of the window
        if loss - orig_loss <= 0:
            return message, False

        norm_orig_loss = (orig_loss - min_loss) / loss_range
        num_positives = 0
        for i in range(1, len(self.losses)):
            norm_loss = (self.losses[i] - min_loss) / loss_range
            if norm_loss > self.magnitude_threshold and norm_loss > norm_orig_loss:
                num_positives += 1

        if num_positives >= self.window_size * self.frequency_threshold:
            min_step = min(self.steps)
            step_range = max(self.steps) - min_step
            steps = (np.array(self.steps) - min_step) / step_range
            losses = (np.array(self.losses) - min_loss) / loss_range
            result = linregress(steps, losses)
            if result.slope > self.slope_threshold:
                message = "Divergence detected"
                return message, True
        return message, False

    def sample(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)