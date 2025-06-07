import collections
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from river import base, stats
from river.utils.random import poisson
from .base_tree_regressor import BaseTreeRegressor
from .arf_regressor import ARFRegressor


class SmartARFRegressor(ARFRegressor):
    def __init__(
        self,
        n_models: int = 10,
        max_models: int = 30,
        regression_pruning_error_threshold: float = 0.1,
        accuracy_drop_threshold: float = 0.5,
        monitor_window: int = 100,
        disable_weighted_vote: bool = False,
        verbose_logging: bool = False,
        min_ensemble_size: int = 5, # Default minimum size
        **kwargs
    ):
        # Ensure n_models used for initialization is at least min_ensemble_size
        effective_n_models = max(n_models, min_ensemble_size)

        super().__init__(n_models=effective_n_models, # ARFRegressor gets effective_n_models
                         disable_weighted_vote=disable_weighted_vote,
                         **kwargs) # Pass other ARF params to ARFRegressor -> BaseForest

        self.max_models = max_models
        # self.n_models inherited from ARFRegressor will be effective_n_models
        self.regression_pruning_error_threshold = regression_pruning_error_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window
        self.verbose_logging = verbose_logging
        self.min_ensemble_size = min_ensemble_size # Store the desired minimum

        self.model_count_history: list[int] = []
        self._accuracy_window: list[collections.deque] = []
        self._warned_tree_ids: set[int] = set()
        self._warning_step: dict[int, int] = {}
        self._warned_recent_acc: dict[int, float] = {}
        self._n_samples_seen = 0
        # Initial log after all super inits have run
        # self._log(f"SmartARF initialized. Effective n_models={self.n_models}, max_models={self.max_models}, min_ensemble_size={self.min_ensemble_size}")
        # For Drift logging
        self.drift_points: list[int] = []
        self.warning_points: list[int] = []

    @property
    def _min_number_of_models(self) -> int:
        """Defines the minimum number of models SmartARF tries to maintain."""
        return self.min_ensemble_size


    def _log(self, message: str):
        if self.verbose_logging:
            # Ensure _n_samples_seen is available, might not be if called from __init__ super chain early
            step_info = getattr(self, '_n_samples_seen', 'init')
            print(f"[SmartARF Step {step_info}] {message}")

    def _init_pruning_state(self): # Called by BaseForest._init_ensemble
        count = len(self.data) # self.data is populated by BaseForest._init_ensemble calling self.append
        self._accuracy_window = [collections.deque(maxlen=self.monitor_window) for _ in range(count)]
        self._warned_tree_ids.clear()
        self._warning_step.clear()
        self._warned_recent_acc.clear()

    def _add_new_model_to_ensemble(self, new_model: BaseTreeRegressor):
        self._log(f"Adding new model. Ensemble size before: {len(self.data)}")
        super()._add_new_model_to_ensemble(new_model) # ARFRegressor's method
        self._accuracy_window.append(collections.deque(maxlen=self.monitor_window))
        self._log(f"Model added. Ensemble size after: {len(self.data)}")

    def _remove_model(self, index_to_remove: int):
        if not (0 <= index_to_remove < len(self.data)):
            return
        self._log(f"Removing model at index {index_to_remove}. Ensemble size before: {len(self.data)}")
        # self.pop is from river.base.Ensemble (UserList)
        # No need to call super(ARFRegressor, self).pop or similar explicitly
        self.pop(index_to_remove)
        lists_to_prune_attrs = [
            '_metrics', '_drift_detectors', '_warning_detectors',
            '_background', '_drift_norm', '_accuracy_window'
        ]
        for attr_name in lists_to_prune_attrs:
            lst = getattr(self, attr_name)
            if index_to_remove < len(lst):
                lst.pop(index_to_remove)

        new_warned_tree_ids = set()
        for warned_idx in self._warned_tree_ids:
            if warned_idx == index_to_remove:
                continue
            elif warned_idx > index_to_remove:
                new_warned_tree_ids.add(warned_idx - 1)
            else:
                new_warned_tree_ids.add(warned_idx)
        self._warned_tree_ids = new_warned_tree_ids

        new_warning_step = {}
        for warned_idx, step_val in self._warning_step.items():
            if warned_idx == index_to_remove:
                continue
            elif warned_idx > index_to_remove:
                new_warning_step[warned_idx - 1] = step_val
            else:
                new_warning_step[warned_idx] = step_val
        self._warning_step = new_warning_step

        new_warned_recent_acc = {}
        for warned_idx, acc_val in self._warned_recent_acc.items():
            if warned_idx == index_to_remove:
                continue
            elif warned_idx > index_to_remove:
                new_warned_recent_acc[warned_idx - 1] = acc_val
            else:
                new_warned_recent_acc[warned_idx] = acc_val
        self._warned_recent_acc = new_warned_recent_acc
        self._log(f"Model removed. Ensemble size after: {len(self.data)}")


    def learn_one(self, x: dict, y: base.typing.Target, **kwargs): # type: ignore
        if not self: # if self.data is empty
            self._init_ensemble(sorted(x.keys()))

        self._n_samples_seen += 1
        step = self._n_samples_seen
        self.model_count_history.append(len(self.data))
        if self._n_samples_seen == 1: # Log initial state after first learn one call
             self._log(f"First learn_one. Effective n_models={self.n_models}, max_models={self.max_models}, min_ensemble_size={self.min_ensemble_size}")
        self._log(f"Start learn_one. Current ensemble size: {len(self.data)}")


        drift_indices = []
        warning_indices = []

        # Stage 1
        for i in range(len(self.data)):
            if i >= len(self.data) or self.data[i] is None: continue
            if i >= len(self._metrics): continue

            model = self.data[i]
            pred = model.predict_one(x)
            self._metrics[i].update(y, pred) # type: ignore
            acc_flag = 1 if abs(float(y) - float(pred)) <= self.regression_pruning_error_threshold else 0
            if i < len(self._accuracy_window):
                 self._accuracy_window[i].append(acc_flag)

            if not self._warning_detection_disabled:
                if i >= len(self._warning_detectors): continue
                wd_input = self._drift_detector_input(i, y, pred)
                self._warning_detectors[i].update(wd_input)
                if self._warning_detectors[i].drift_detected:
                    self._log(f"Warning detected for model {i}.")
                    warning_indices.append(i)
                    self.warning_points.append(step)

            if not self._drift_detection_disabled:
                if i >= len(self._drift_detectors): continue
                dd_input = self._drift_detector_input(i, y, pred)
                self._drift_detectors[i].update(dd_input)
                if self._drift_detectors[i].drift_detected:
                    self._log(f"Drift detected for model {i}.")
                    drift_indices.append(i)
                    self.drift_points.append(step)

        # Stage 2
        for i in warning_indices:
            if i >= len(self._background) or i >= len(self._warning_detectors) : continue
            if self._background[i] is None:
                self._log(f"Creating background learner for warned model {i}.")
                self._background[i] = self._new_base_model()
            self._warning_detectors[i] = self.warning_detector.clone()
            self._warning_tracker[i] += 1
            if i not in self._warned_tree_ids:
                self._warned_tree_ids.add(i)
                self._warning_step[i] = step
                if i < len(self._accuracy_window) and len(self._accuracy_window[i]) >= self.monitor_window:
                    past_acc = sum(self._accuracy_window[i]) / len(self._accuracy_window[i])
                    self._warned_recent_acc[i] = past_acc
                    self._log(f"Model {i} warned. Storing past accuracy: {past_acc:.3f} over {len(self._accuracy_window[i])} samples.")
                else:
                    self._warned_recent_acc[i] = 1.0 # Default if not enough data
                    self._log(f"Model {i} warned. Not enough samples ({len(self._accuracy_window[i]) if i < len(self._accuracy_window) else 'N/A'}) for past accuracy, defaulting to 1.0.")


        # Stage 3
        background_models_to_add = []
        drifted_indices_processed_for_reset = set()

        for i in sorted(list(set(drift_indices))):
            if i >= len(self.data): continue
            if i in drifted_indices_processed_for_reset: continue

            self._log(f"Processing drift for model {i}. Current bg: {'Exists' if i < len(self._background) and self._background[i] else 'None'}")
            self._drift_tracker[i] += 1
            if i < len(self._background) and self._background[i]:
                self._log(f"Marking background model from slot {i} to be added.")
                background_models_to_add.append(self._background[i])
                self._background[i] = None

            self._log(f"Resetting drifted model {i}.")
            self.data[i] = self._new_base_model()
            if i < len(self._metrics): self._metrics[i] = self.metric.clone()
            if i < len(self._drift_detectors): self._drift_detectors[i] = self.drift_detector.clone()
            if not self._warning_detection_disabled and i < len(self._warning_detectors):
                self._warning_detectors[i] = self.warning_detector.clone()
            if i < len(self._drift_norm): self._drift_norm[i] = stats.Var()
            if i < len(self._accuracy_window): self._accuracy_window[i].clear()

            if i in self._warned_tree_ids:
                self._warned_tree_ids.discard(i)
                self._warning_step.pop(i, None)
                self._warned_recent_acc.pop(i, None)
            drifted_indices_processed_for_reset.add(i)

        for bg_model_candidate in background_models_to_add:
            if len(self.data) < self.max_models:
                self._log(f"Adding promoted background model. Ensemble size < max_models ({len(self.data)} < {self.max_models}).")
                self._add_new_model_to_ensemble(bg_model_candidate)
            else:
                self._log(f"Ensemble at max_models ({self.max_models}). Finding worst model to replace for promoted BG.")
                worst_model_idx = self._find_worst_model()
                if worst_model_idx is not None:
                    self._log(f"Worst model is {worst_model_idx}. Removing it to make space.")
                    self._remove_model(worst_model_idx) # This might make len(self.data) < max_models
                    self._log(f"Adding promoted background model after removing worst.")
                    self._add_new_model_to_ensemble(bg_model_candidate)
                else:
                    self._log("Could not find a worst model to replace, promoted background model not added.")

        while len(self.data) > self.max_models:
            self._log(f"Ensemble size ({len(self.data)}) > max_models ({self.max_models}). Pruning worst.")
            worst_model_idx = self._find_worst_model()
            # Also check if we are already at or below the defined minimum
            if worst_model_idx is not None and len(self.data) > self._min_number_of_models:
                self._remove_model(worst_model_idx)
            else:
                self._log(f"Cannot prune further (no worst found or at/below min models {self._min_number_of_models}).")
                break

        # Stage 4
        indices_to_remove_due_to_acc_drop = []
        current_warned_ids = list(self._warned_tree_ids)
        for i in current_warned_ids:
            if not (0 <= i < len(self.data)):
                self._warned_tree_ids.discard(i)
                self._warning_step.pop(i, None)
                self._warned_recent_acc.pop(i, None)
                continue

            age_since_warning = step - self._warning_step.get(i, step)
            if i >= len(self._accuracy_window): continue
            current_acc_window = self._accuracy_window[i]

            if len(current_acc_window) < self.monitor_window:
                if age_since_warning > 2 * self.monitor_window:
                    self._log(f"Warned model {i} has insufficient data for acc drop check for too long. Clearing warning state.")
                    self._warned_tree_ids.discard(i)
                    self._warning_step.pop(i, None)
                    self._warned_recent_acc.pop(i, None)
                continue

            current_accuracy = sum(current_acc_window) / len(current_acc_window)
            accuracy_at_warning_time = self._warned_recent_acc.get(i, 1.0)
            self._log(f"AccDropCheck for warned model {i}: Age {age_since_warning}, CurrentAcc {current_accuracy:.3f} (win {len(current_acc_window)}), PastAcc {accuracy_at_warning_time:.3f}, Thr {self.accuracy_drop_threshold * accuracy_at_warning_time:.3f}")

            if accuracy_at_warning_time > 1e-6 and \
               current_accuracy < (self.accuracy_drop_threshold * accuracy_at_warning_time):
                self._log(f"Sustained accuracy drop for model {i}. Marking for removal.")
                indices_to_remove_due_to_acc_drop.append(i)

        for i in sorted(indices_to_remove_due_to_acc_drop, reverse=True):
            if len(self.data) > self._min_number_of_models: # Use the property here
                self._log(f"Pruning model {i} due to accuracy drop.")
                self._remove_model(i)
            else:
                self._log(f"Skipping pruning model {i} (accuracy drop) because at min models ({self._min_number_of_models}).")

        # Stage 5
        for i in range(len(self.data)):
            if i >= len(self.data) or self.data[i] is None : continue
            k = poisson(self.lambda_value, rng=self._rng)
            if k <= 0: continue
            if not self._warning_detection_disabled:
                if i < len(self._background) and self._background[i] is not None:
                    self._background[i].learn_one(x, y, w=k) # type: ignore
            if i < len(self.data) and self.data[i] is not None:
                 self.data[i].learn_one(x, y, w=k)
        self._log(f"End learn_one. Final ensemble size: {len(self.data)}")
        return self

    def _find_worst_model(self) -> int | None:
        if not self.data or not self._metrics:
            return None
        valid_metrics = []
        valid_indices = []
        for i in range(min(len(self.data), len(self._metrics))):
            metric_obj = self._metrics[i]
            try:
                val = metric_obj.get()
                if not (math.isnan(val) or math.isinf(val)):
                    valid_metrics.append(val)
                    valid_indices.append(i)
            except Exception:
                pass
        if not valid_metrics:
            return None if not self.data else random.randrange(len(self.data)) if self.data else None

        if self.metric.bigger_is_better: # type: ignore
            worst_idx_in_valids = np.argmin(valid_metrics)
        else:
            worst_idx_in_valids = np.argmax(valid_metrics)
        return valid_indices[worst_idx_in_valids]

    def plot_model_count(self, mark_warnings: bool = False, mark_drifts: bool = False):
        if not self.model_count_history:
            print("No model count history to plot for SmartARFRegressor.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.model_count_history, drawstyle="steps-post", label="Actual Ensemble Size")

        plt.axhline(self.max_models, color='r', linestyle='--', label=f"Max Models ({self.max_models})")
        plt.axhline(self.n_models, color='g', linestyle=':', label=f"Initial/Target Models ({self.n_models})")
        if self._min_number_of_models > 0:
            plt.axhline(self._min_number_of_models, color='b', linestyle=':', label=f"Min Ensemble Size ({self._min_number_of_models})")

        # Add vertical drift and warning markers
        if mark_warnings and hasattr(self, "warning_points"):
            for step in self.warning_points:
                plt.axvline(x=step, color='orange', linestyle='--', alpha=0.4, label='Drift Warning')
        if mark_drifts and hasattr(self, "drift_points"):
            for step in self.drift_points:
                plt.axvline(x=step, color='red', linestyle='-', alpha=0.5, label='Drift Detected')

        # Avoid duplicate legend labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlabel("Instances Processed (Learn One Calls)")
        plt.ylabel("Number of Models in Ensemble")
        plt.title("SmartARFRegressor Ensemble Size Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

