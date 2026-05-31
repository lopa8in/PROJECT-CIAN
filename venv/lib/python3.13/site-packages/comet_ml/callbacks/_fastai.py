# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
import tempfile
from typing import Any, Dict

import comet_ml
from comet_ml import convert_utils

from fastai import basics, learner
from fastai.callback import hook

LOGGER = logging.getLogger(__name__)


class CometFastAICallback(learner.Callback):
    "Log losses, metrics, model weights, model architecture summary to comet"
    order = learner.Recorder.order + 1

    def __init__(
        self, experiment: comet_ml.CometExperiment, log_model_to_experiment: bool = True
    ):
        self.experiment = experiment
        self.log_model_to_experiment = log_model_to_experiment

    def before_fit(self):
        self._comet_step = (
            self.experiment.curr_step if self.experiment.curr_step is not None else 0
        )
        if self.experiment.auto_param_logging:
            self._log_properties()
            self._log_parameters()
        self._log_model_summary()

    def after_batch(self):
        if not self.training:
            return

        self.experiment.set_step(self._comet_step)
        if self.experiment.auto_metric_logging:
            try:
                self.experiment.log_metric(
                    "batch__smooth_loss",
                    value=convert_utils.try_move_to_cpu(self.smooth_loss),
                )
                self.experiment.log_metric(
                    "batch__loss", value=convert_utils.try_move_to_cpu(self.loss)
                )
                self.experiment.log_metric("batch__train_iter", value=self.train_iter)
                for hypers in self.opt.hypers:
                    for hyper_key, hyper_value in hypers.items():
                        self.experiment.log_metric(
                            f"batch__opt.hypers.{hyper_key}", value=hyper_value
                        )
            except Exception:
                LOGGER.warning("Failed to log batch metrics", exc_info=True)

        self._comet_step += 1

    def after_epoch(self):
        self.experiment.set_epoch(self.epoch)

        if not self.experiment.auto_metric_logging:
            return

        try:
            for number, value in zip(self.recorder.metric_names, self.recorder.log):
                if number not in ["epoch", "time"]:
                    self.experiment.log_metric(f"epoch__{number}", value)
        except Exception:
            LOGGER.warning("Failed to log epoch metrics to Comet", exc_info=True)

    def after_fit(self):
        if not (self.log_model_to_experiment and hasattr(self.learn, "save")):
            return

        try:
            with tempfile.NamedTemporaryFile(mode="w") as model_file:
                model_path = self.learn.save(model_file.name)
                self.experiment.log_model(
                    name="FastAI_model",
                    file_or_folder=str(model_path),
                    file_name="model",
                )
        except Exception:
            LOGGER.warning("Failed to log the model to Comet", exc_info=True)

    def _log_properties(self):
        try:
            self.experiment.log_parameter("n_epoch", str(self.n_epoch))
            self.experiment.log_parameter("model_class", str(type(self.model)))
        except Exception:
            LOGGER.warning("Failed to log properties to Comet", exc_info=True)

    def _log_parameters(self):
        config_parameters = extract_parameters(self.learn)

        try:
            self.experiment.log_parameters(config_parameters)
        except Exception:
            LOGGER.warning("Failed to log parameters to Comet", exc_info=True)

    def _log_model_summary(self):
        try:
            with tempfile.NamedTemporaryFile(mode="w") as summary_file:
                summary_file.write(repr(self.model))
                summary_file.seek(0)
                self.experiment.log_asset(summary_file.name, "model_summary.txt")
        except Exception:
            LOGGER.warning("Failed to log the model summary to Comet", exc_info=True)


def extract_parameters(learner: learner.Learner) -> Dict[str, Any]:
    "Gather config parameters accessible to the learner"
    args = {}

    try:
        learner_parameters = {}
        learner_parameters["name"] = learner
        learner_parameters["arch"] = learner.arch
        learner_parameters["n_out"] = learner.n_out
        learner_parameters["normalize"] = learner.normalize
        learner_parameters["opt_func"] = learner.opt
        learner_parameters["pretrained"] = learner.pretrained
        learner_parameters["splitter"] = learner.splitter
        learner_parameters["train_bn"] = learner.train_bn
        learner_parameters["wd"] = learner.wd
        learner_parameters["wd_bn_bias"] = learner.wd_bn_bias

        args["Learner"] = learner_parameters

    except Exception:
        LOGGER.warning("Failed to log Learner parameters to Comet")

    try:
        n_inp = learner.dls.train.n_inp
        args["n_inp"] = n_inp
        xb = learner.dls.valid.one_batch()[:n_inp]
        args.update(
            {
                f"input {n + 1} dim {i + 1}": d
                for n in range(n_inp)
                for i, d in enumerate(list(basics.detuplify(xb[n]).shape))
            }
        )
    except Exception:
        LOGGER.warning("Could not gather input dimensions")

    with basics.ignore_exceptions():
        args["batch_size"] = learner.dls.bs
        args["batch_per_epoch"] = len(learner.dls.train)
        args["model_parameters"] = hook.total_params(learner.model)[0]
        args["device"] = learner.dls.device.type
        args["frozen"] = bool(learner.opt.frozen_idx)
        args["frozen idx"] = learner.opt.frozen_idx
        args["dataset.tfms"] = f"{learner.dls.dataset.tfms}"
        args["dls.after_item"] = f"{learner.dls.after_item}"
        args["dls.before_batch"] = f"{learner.dls.before_batch}"
        args["dls.after_batch"] = f"{learner.dls.after_batch}"
    return args
