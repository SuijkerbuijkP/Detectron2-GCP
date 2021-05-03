import datetime
import logging
import time

import detectron2.utils.comm as comm
import hypertune
import numpy as np
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage, EventWriter
from detectron2.utils.logger import log_every_n_seconds


class LossMetricWriter(EventWriter):
    """
    Reports a single loss metric to CloudML Hyptertune to accommodate hyperparameter training. This is done every 20
    iterations, which is equal to the hardcoded writer period.
    """

    def __init__(self, window_size: int = 20):
        self.logger = logging.getLogger(__name__)
        self._window_size = window_size
        self.hpt = hypertune.HyperTune()

    def write(self):
        storage = get_event_storage()
        loss = storage.histories().get("total_loss")
        if loss is not None:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='total_loss',
                metric_value=loss.median(self._window_size),
                global_step=storage.iter
            )


class LossEvalHook(HookBase):
    """
    Taken from https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e.
    This hook computes validation loss at EVAL_PERIOD or at the end of training.

    As an improvement, this could maybe be fused with EvalHook, as now the validation dataset is passed twice every period.
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.pyl
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
