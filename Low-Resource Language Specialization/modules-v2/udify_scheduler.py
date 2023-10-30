"""
Special LR scheduler for fine-tuning Transformer networks
"""

from overrides import overrides
import torch
import logging

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import (
    LearningRateScheduler,
)

logger = logging.getLogger(__name__)


@LearningRateScheduler.register("ulmfit_sqrt")
class UlmfitSqrtLR(LearningRateScheduler):
    """
    Implements a combination of ULMFiT (slanted triangular) with Noam sqrt
    learning rate decay
    from https://github.com/Hyperparticle/udify/blob/master/udify/optimizers/ulmfit_sqrt.py
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int,
        affected_group_count: int,
        start_step: int = 0,
        factor: float = 100,
        steepness: float = 0.5,
        last_epoch: int = -1,
        gradual_unfreezing: bool = False,
        discriminative_fine_tuning: bool = False,
        decay_factor: float = 0.38,
    ) -> None:
        self.warmup_steps = warmup_steps + start_step
        self.start_step = start_step
        self.factor = factor
        self.steepness = steepness
        self.model_size = model_size
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing
        self.affected_group_count = affected_group_count

        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1][
                "params"
            ], "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert (
                len(optimizer.param_groups) >= 2 * affected_group_count + 1
            ), (
                "There should be at least affected_group_count + 2 param_groups"
                " (discriminative groups + control groups + empty default group)"
                " for gradual unfreezing / discriminative fine-tuning to make sense."
            )

        super().__init__(optimizer, last_epoch=last_epoch)

        if discriminative_fine_tuning:
            # exponent = 0
            # for i in range(len(self.base_values) - 1, -1, -1):
            #     param_group = optimizer.param_groups[i]
            #     # skip the last param_group if it is has no parameters
            #     if param_group["params"]:
            #         param_group["lr"] = (
            #             self.base_values[i] * decay_factor ** exponent
            #         )
            #         self.base_values[i] = param_group["lr"]
            #         exponent += 1

            # only the first `self.affected_group_count` groups are affected
            # NOTE: this could be made more general (for arbitrary tuples of
            # size `self.affected_group_count`, but it's not necessary here)
            for i in range(self.affected_group_count):
                param_group = optimizer.param_groups[i]
                assert param_group[
                    "params"
                ], "Tried to discriminatively fine-tune an empty group"
                param_group["lr"] = self.base_values[i] * decay_factor
                self.base_values[i] = param_group["lr"]

        # AllenNLP used to call step prior to initializing the first epoch (see
        # comments below) but now only calls it after the first epoch.  So in
        # order to make sure everything is set up properly, we call it and
        # manually (hackily) track the epoch
        self.is_first_epoch = True
        self.step()

    @overrides
    def step(self, metric: float = None) -> None:
        if self.gradual_unfreezing:
            # we only have one set of things to unfreeze, and that will always
            # happen AFTER the first epoch
            if self.is_first_epoch:
                logger.info(
                    "Gradual unfreezing in progress for first "
                    f"{self.affected_group_count} groups"
                )
            else:
                logger.info("Gradual unfreezing done; training all groups")

            for i, param_group in enumerate(
                reversed(self.optimizer.param_groups)
            ):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    # the lower `self.affected_group_count` are the controls
                    param.requires_grad = (not self.is_first_epoch) or bool(
                        i <= self.affected_group_count
                    )
            self.is_first_epoch = False

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = batch_num_total
        for param_group, learning_rate in zip(
            self.optimizer.param_groups, self.get_values()
        ):
            param_group["lr"] = learning_rate

    def get_values(self):
        if self.freezing_current:
            # If parameters are still frozen, keep the base learning rates
            return self.base_values

        # This computes the Noam Sqrt LR decay based on the current step
        step = max(self.last_epoch - self.start_step, 1)
        scale = self.factor * (
            self.model_size ** (-0.5)
            * min(
                step ** (-self.steepness),
                step * self.warmup_steps ** (-self.steepness - 1),
            )
        )

        return [scale * lr for lr in self.base_values]
