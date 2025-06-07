from __future__ import annotations

import math
from river import base, stats
from river.utils.random import poisson
from typing import Literal
import numpy as np
from .arf_regressor import ARFRegressor


class ARFRegressorDynamicWeights(ARFRegressor):
    """
    Adaptive Random Forest Regressor with dynamic model weighting.

    This regressor adjusts weights of individual trees based on their prediction performance,
    using either a moving Mean Absolute Error (MAE) or the standard deviation of errors (STD).

    Parameters
    ----------
    error_mode : {'mae', 'std'}, default='std'
        Strategy for error comparison to update tree scores.
    error_threshold_factor : float, default=1.0
        Tolerance multiplier for determining "good" predictions.
    """

    def __init__(
        self,
        *args,
        error_mode: Literal["mae", "std"] = "std",
        error_threshold_factor: float = 1.0,
        **kwargs,
    ):
        kwargs["disable_weighted_vote"] = True
        kwargs["aggregation_method"] = self._MEAN  # Enforce mean

        super().__init__(*args, **kwargs)
        self.error_mode = error_mode
        self.error_threshold_factor = error_threshold_factor

        self._dynamic_perf_scores: list[float] = []
        self._dynamic_weights: list[float] = []

        if self.error_mode == "mae":
            self._mean_abs_errors: list[stats.Mean] = []
        if len(self) > 0:
            self._init_dynamic_weights()

    def _init_dynamic_weights(self):
        self._dynamic_perf_scores = [1.0] * self.n_models
        equal_weight = 1.0 / self.n_models if self.n_models > 0 else 1.0
        self._dynamic_weights = [equal_weight] * self.n_models

        if self.error_mode == "mae":
            self._mean_abs_errors = [stats.Mean() for _ in range(self.n_models)]

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        self._init_dynamic_weights()

    def _update_dynamic_weights(self):
        raw_weights = [1.0 / (1.0 + score) for score in self._dynamic_perf_scores]
        total = sum(raw_weights)
        if total > 0:
            self._dynamic_weights = [w / total for w in raw_weights]
        else:
            fallback = 1.0 / self.n_models if self.n_models > 0 else 1.0
            self._dynamic_weights = [fallback] * self.n_models

    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs):
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        tree_predictions = [0.0] * self.n_models

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)
            tree_predictions[i] = y_pred
            abs_error = abs(y - y_pred)
            self._metrics[i].update(y_true=y, y_pred=y_pred)

            if self.error_mode == "mae":
                self._mean_abs_errors[i].update(abs_error)
                mae = self._mean_abs_errors[i].get()
                threshold = (
                    (1 + self.error_threshold_factor) * mae if mae > 1e-9 else 0.01
                )
            else:  # "std"
                std = (
                    math.sqrt(self._drift_norm[i].get())
                    if self._drift_norm[i].mean.n > 1
                    else float("inf")
                )
                threshold = self.error_threshold_factor * std

            if abs_error <= threshold:
                self._dynamic_perf_scores[i] *= 0.9
            else:
                self._dynamic_perf_scores[i] *= 1.1

            self._dynamic_perf_scores[i] = max(0.01, min(self._dynamic_perf_scores[i], 100.0))

        for i, model in enumerate(self):
            drift_input = None
            if not self._warning_detection_disabled:
                drift_input = self._drift_detector_input(i, y, tree_predictions[i])
                self._warning_detectors[i].update(drift_input)
                if self._warning_detectors[i].drift_detected:
                    if self._background:
                        self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    self._warning_tracker[i] += 1

            if not self._drift_detection_disabled:
                drift_input = (
                    drift_input or self._drift_detector_input(i, y, tree_predictions[i])
                )
                self._drift_detectors[i].update(drift_input)
                if self._drift_detectors[i].drift_detected:
                    self.data[i] = (
                        self._background[i]
                        if self._background and self._background[i] is not None
                        else self._new_base_model()
                    )
                    self._background[i] = None
                    self._warning_detectors[i] = self.warning_detector.clone()
                    self._drift_detectors[i] = self.drift_detector.clone()
                    self._metrics[i] = self.metric.clone()
                    self._drift_norm[i] = stats.Var()
                    self._dynamic_perf_scores[i] = 1.0
                    if self.error_mode == "mae":
                        self._mean_abs_errors[i] = stats.Mean()
                    self._drift_tracker[i] += 1

        self._update_dynamic_weights()

        for i, model in enumerate(self):
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                if (
                    not self._warning_detection_disabled
                    and self._background
                    and self._background[i] is not None
                ):
                    self._background[i].learn_one(x=x, y=y, w=k)
                model.learn_one(x=x, y=y, w=k)
        return self

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return 0.0

        if not self._dynamic_weights or len(self._dynamic_weights) != self.n_models:
            self._init_dynamic_weights()

        preds = np.array([model.predict_one(x) for model in self])
        weights = np.array(self._dynamic_weights)

        if self.aggregation_method == self._MEAN:
            return float(np.dot(preds, weights))
        elif self.aggregation_method == self._MEDIAN:
            return float(np.median(preds))
        else:
            return float(np.mean(preds))
