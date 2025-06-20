from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
import numpy as np
import math
from river.utils.random import poisson
from river.tree.splitter import Splitter
from .base_tree_regressor import BaseTreeRegressor
from .base_forest import BaseForest


class ARFRegressor(BaseForest, base.Regressor):
    _MEAN = "mean"
    _MEDIAN = "median"

    def __init__(
        self,
        n_models: int = 10,
        max_features="sqrt",
        aggregation_method: str = "mean",
        lambda_value: int = 6,
        metric: metrics.base.RegressionMetric | None = None,
        disable_weighted_vote: bool = True,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        grace_period: int = 50,
        max_depth: int | None = None,
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 500.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__( # This now calls the modified BaseForest.__init__
            n_models=n_models,
            max_features=max_features,
            lambda_value=lambda_value,
            metric=metric if metric is not None else metrics.MSE(),
            disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector if drift_detector is not None else ADWIN(0.001),
            warning_detector=warning_detector if warning_detector is not None else ADWIN(0.01),
            seed=seed,
        )
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.leaf_model = leaf_model
        self.model_selector_decay = model_selector_decay
        self.nominal_attributes = nominal_attributes
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.binary_split = binary_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError(f"Invalid aggregation_method: {aggregation_method}")
        self.aggregation_method = aggregation_method
        self._drift_norm: list[stats.Var] = []

    def _init_drift_norm(self):
        self._drift_norm = [stats.Var() for _ in range(len(self.data))]

    @property
    def _mutable_attributes(self):
        return {"max_features", "aggregation_method", "lambda_value",
                "grace_period", "max_depth", "delta", "tau", "leaf_prediction",
                "leaf_model", "model_selector_decay", "nominal_attributes",
                "splitter", "min_samples_split", "binary_split", "max_size",
                "memory_estimate_period", "stop_mem_management",
                "remove_poor_attrs", "merit_preprune"}

    def _new_base_model(self) -> BaseTreeRegressor:
        return BaseTreeRegressor(
            max_features=self.max_features, # type: ignore
            grace_period=self.grace_period,
            max_depth=self.max_depth,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            leaf_model=self.leaf_model,
            model_selector_decay=self.model_selector_decay,
            nominal_attributes=self.nominal_attributes,
            splitter=self.splitter,
            min_samples_split=self.min_samples_split,
            binary_split=self.binary_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            rng=self._rng,
        )

    def _drift_detector_input(
        self, tree_id: int, y_true: base.typing.Target, y_pred: base.typing.RegTarget # type: ignore
    ) -> float:
        y_t = float(y_true)
        y_p = float(y_pred)
        error = y_t - y_p
        if tree_id >= len(self._drift_norm):
            return 0.5
        self._drift_norm[tree_id].update(error)

        if self._drift_norm[tree_id].mean.n <= 1: # type: ignore
            return 0.5

        variance = self._drift_norm[tree_id].get()
        if variance < 1e-9:
             return 0.0 if abs(error) < 1e-9 else 1.0

        sd = math.sqrt(variance)
        normalized_error = abs(error / (3 * sd)) if sd > 0 else (1.0 if error != 0 else 0.0)
        return min(1.0, max(0.0, normalized_error))

    def _add_new_model_to_ensemble(self, new_model: BaseTreeRegressor):
        self.append(new_model)
        self._metrics.append(self.metric.clone())
        if not self._drift_detection_disabled:
            self._drift_detectors.append(self.drift_detector.clone())
        if not self._warning_detection_disabled:
            self._warning_detectors.append(self.warning_detector.clone())
            self._background.append(None)
        self._drift_norm.append(stats.Var())

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs): # type: ignore
        if not self:
            self._init_ensemble(sorted(x.keys()))

        for i in range(len(self.data)):
            model = self.data[i]
            y_pred = model.predict_one(x)
            self._metrics[i].update(y, y_pred) # type: ignore
            k = poisson(self.lambda_value, rng=self._rng)
            if k <= 0:
                continue

            if not self._warning_detection_disabled and self._background[i] is not None:
                self._background[i].learn_one(x, y, w=k) # type: ignore

            if not self._warning_detection_disabled:
                wd_input = self._drift_detector_input(i, y, y_pred)
                self._warning_detectors[i].update(wd_input)
                if self._warning_detectors[i].drift_detected:
                    if self._background[i] is None:
                        self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    self._warning_tracker[i] += 1

            if not self._drift_detection_disabled:
                dd_input = self._drift_detector_input(i, y, y_pred)
                self._drift_detectors[i].update(dd_input)
                if self._drift_detectors[i].drift_detected:
                    self._drift_tracker[i] += 1
                    current_background_learner = self._background[i]

                    if current_background_learner is not None:
                        self.data[i] = current_background_learner
                        self._background[i] = self._new_base_model()
                    else:
                        self.data[i] = self._new_base_model()
                        self._background[i] = None

                    self._metrics[i] = self.metric.clone()
                    self._drift_detectors[i] = self.drift_detector.clone()
                    if not self._warning_detection_disabled:
                         self._warning_detectors[i] = self.warning_detector.clone()
                    self._drift_norm[i] = stats.Var()
                    self._warning_tracker[i] = 0
            model.learn_one(x, y, w=k)
        return self

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if not self:
            self._init_ensemble(sorted(x.keys()))

        if not self.data:
            return 0.0

        preds = np.array([m.predict_one(x) for m in self.data if m is not None])
        if preds.size == 0:
            if self.n_models > 0 :
                pass
            return 0.0

        if self.aggregation_method == self._MEDIAN:
            return float(np.median(preds))

        if self.disable_weighted_vote or len(self.data) <= 1: # type: ignore
            return float(np.mean(preds))
        else:
            weights = []
            valid_preds = []
            for idx in range(len(self.data)):
                if idx >= len(preds) or idx >= len(self._metrics):
                    continue
                pred_val = preds[idx]
                metric_obj = self._metrics[idx]
                try:
                    metric_val = metric_obj.get()
                    if not (math.isnan(metric_val) or math.isinf(metric_val)):
                        weights.append(metric_val)
                        valid_preds.append(pred_val)
                except Exception:
                    pass

            if not weights or not valid_preds:
                return float(np.mean(preds))

            weights_arr = np.array(weights)
            valid_preds_arr = np.array(valid_preds)

            if not self.metric.bigger_is_better: # type: ignore
                max_val = np.max(weights_arr)
                if max_val > 1e-9:
                    weights_arr = max_val - weights_arr
                else:
                    weights_arr = np.ones_like(weights_arr)

            sum_weights = np.sum(weights_arr)
            if sum_weights <= 1e-9:
                return float(np.mean(valid_preds_arr))
            return float(np.dot(valid_preds_arr, weights_arr) / sum_weights)