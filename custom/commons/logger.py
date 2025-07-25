from typing import Literal
import neptune
from neptune.utils import stringify_unsupported
from custom.models.saint_transformer.config import SAINTConfig
import os


class McLogger:
    def __init__(self, config: SAINTConfig):
        neptune_run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        neptune_run["config"] = stringify_unsupported(config.__dict__)

        self.neptune_run: neptune.Run = neptune_run
        self.config: SAINTConfig = config
        self.global_step: int = 0
        self._context: Literal["train", "val", "meta"] = "train"

    def set_context(self, context: Literal["train", "val", "meta"]):
        self._context = context

    def log(self, name, value, on_step=None, on_epoch=None):
        if self._context == "train":
            should_log = on_step if on_step is not None else True
            if should_log and self.global_step % self.config.log_every_n_steps == 0:
                self._write_log(f"{self._context}/{name}", value)

        elif self._context == "val":
            should_log = on_epoch if on_epoch is not None else True
            if should_log:
                self._write_log(f"{self._context}/{name}", value)

        elif self._context == "meta":
            should_log = on_epoch if on_epoch is not None else True
            if should_log:
                self._write_log(f"{self._context}/{name}", value)

    def _write_log(self, key, value):
        if self.neptune_run:
            self.neptune_run[key].log(value, step=self.global_step)

    def step(self):
        self.global_step += 1

    def stop(self):
        self.neptune_run.stop()
