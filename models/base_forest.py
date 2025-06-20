from __future__ import annotations

import abc
import collections
import math
import random

from river import base, metrics
from river.drift import NoDrift
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor


class BaseForest(base.Ensemble):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"

    def __init__(
        self,
        n_models: int,
        max_features: bool | str | int,
        lambda_value: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
        disable_weighted_vote: bool,
        seed: int | None,
    ):
        # Calls collections.UserList.__init__ via base.Ensemble's MRO,
        # effectively self.data = []. This bypasses river.base.Ensemble's __init__ logic
        # that performs the _min_number_of_models check on the passed models list.
        super(base.Ensemble, self).__init__([])

        # Initialize BaseForest specific attributes
        self.n_models = n_models # This is the target/initial number of models for _init_ensemble
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.metric = metric
        self.disable_weighted_vote = disable_weighted_vote
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed
        self._rng = random.Random(self.seed)

        self._warning_detectors: list[base.DriftDetector] = []
        self._warning_detection_disabled = isinstance(self.warning_detector, NoDrift)

        self._drift_detectors: list[base.DriftDetector] = []
        self._drift_detection_disabled = isinstance(self.drift_detector, NoDrift)

        self._background: list[HoeffdingTreeClassifier | HoeffdingTreeRegressor | None] = []
        self._metrics: list[metrics.base.Metric] = []

        self._warning_tracker: dict[int, int] = collections.defaultdict(int) if not self._warning_detection_disabled else {}
        self._drift_tracker: dict[int, int] = collections.defaultdict(int) if not self._drift_detection_disabled else {}
        # Models are added via _init_ensemble, typically called on first learn/predict

    @property
    def _min_number_of_models(self) -> int:
        # This is the fallback minimum if a subclass doesn't override it.
        # SmartARFRegressor will override this with its specific minimum.
        return 1

    @abc.abstractmethod
    def _drift_detector_input(self, tree_id: int, y_true, y_pred) -> int | float:
        raise NotImplementedError

    @abc.abstractmethod
    def _new_base_model(self) -> HoeffdingTreeClassifier | HoeffdingTreeRegressor:
        raise NotImplementedError

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        self.data.clear()
        self._metrics.clear()
        self._warning_detectors.clear()
        self._drift_detectors.clear()
        self._background.clear()

        if hasattr(self, '_drift_norm') and isinstance(getattr(self, '_drift_norm'), list):
            getattr(self, '_drift_norm').clear()
        if hasattr(self, '_accuracy_window') and isinstance(getattr(self, '_accuracy_window'), list):
            getattr(self, '_accuracy_window').clear()

        for i in range(self.n_models):
            self.append(self._new_base_model())

        count = len(self.data)
        if not self._warning_detection_disabled:
            self._warning_detectors = [self.warning_detector.clone() for _ in range(count)]
            self._background = [None] * count
        if not self._drift_detection_disabled:
            self._drift_detectors = [self.drift_detector.clone() for _ in range(count)]
        self._metrics = [self.metric.clone() for _ in range(count)]

        if hasattr(self, '_init_drift_norm') and callable(getattr(self, '_init_drift_norm', None)):
            self._init_drift_norm()
        if hasattr(self, '_init_pruning_state') and callable(getattr(self, '_init_pruning_state', None)):
            self._init_pruning_state()

        self._warning_tracker.clear()
        self._drift_tracker.clear()

    def _set_max_features(self, n_features: int):
        orig = self.max_features
        if self.max_features == self._FEATURES_SQRT:
            self.max_features = round(math.sqrt(n_features))
        elif self.max_features == self._FEATURES_LOG2:
            self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, float): # type: ignore
            self.max_features = int(self.max_features * n_features)
        elif self.max_features is None:
            self.max_features = n_features
        elif not isinstance(self.max_features, int):
             raise AttributeError(f"Invalid max_features: {orig}")

        if self.max_features < 0 :
            self.max_features += n_features
        self.max_features = max(1, min(self.max_features, n_features)) # type: ignore

    def n_warnings_detected(self, tree_id: int | None = None) -> int:
        if self._warning_detection_disabled:
            return 0
        if tree_id is None:
            return sum(self._warning_tracker.values())
        return self._warning_tracker.get(tree_id, 0)

    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        if self._drift_detection_disabled:
            return 0
        if tree_id is None:
            return sum(self._drift_tracker.values())
        return self._drift_tracker.get(tree_id, 0)