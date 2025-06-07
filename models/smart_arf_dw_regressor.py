import collections
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from .arf_regressor_dw import ARFRegressorDynamicWeights
from river import base, stats
from river.utils.random import poisson
from .base_tree_regressor import BaseTreeRegressor
from .smart_arf_dw_regressor import ARFRegressorDynamicWeights


class SmartARFDynamicWeightsRegressor(ARFRegressorDynamicWeights):
    """
    Adaptive Random Forest Regressor that combines:
    1. Dynamic tree weighting (adapted 0.9/1.1 rule for regression).
    2. Dynamic ensemble size management (adding trees on drift, pruning on
       proxy "accuracy" drop or exceeding max_models).
    """

    def __init__(
        self,
        n_models: int = 10,
        max_models: int = 30,
        regression_pruning_error_threshold: float = 0.1,  # Absolute error for "accurate"
        accuracy_drop_threshold: float = 0.5,  # Pruning if proxy acc drops by this factor
        monitor_window: int = 100,
        **kwargs,  # Pass other ARFRegressorDynamicWeights params
    ):
        super().__init__(n_models=n_models, **kwargs)  # Pass n_models for initial setup
        self.max_models = max_models
        self.regression_pruning_error_threshold = regression_pruning_error_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window

        self.model_count_history: list[int] = []  # To plot ensemble size
        self._accuracy_window: list[collections.deque] = (
            []
        )  # For pruning based on acc drop
        self._warned_tree_ids: set[int] = set()
        self._warning_step: dict[int, int] = {}  # Instance step when warning started
        self._warned_recent_acc: dict[int, float] = {}  # Accuracy at time of warning

        if len(self.data) > 0:  # If ensemble already exists (e.g. from loading)
            self._init_pruning_state()

    def _init_pruning_state(self):
        num_models = len(self.data)
        # Accuracy window stores 0s and 1s (1 if |error| <= threshold)
        self._accuracy_window = [
            collections.deque(maxlen=self.monitor_window) for _ in range(num_models)
        ]
        self._warned_tree_ids.clear()
        self._warning_step.clear()
        self._warned_recent_acc.clear()

    def _init_ensemble(self, features: list):
        # Call parent's _init_ensemble (ARFRegressorDynamicWeights -> ARFRegressor)
        super()._init_ensemble(features)
        # Initialize smart pruning specific states
        self._init_pruning_state()

    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs):
        if len(self.data) == 0:
            self._init_ensemble(sorted(x.keys()))

        current_step = sum(
            model.total_weight_observed for model in self.data
        )  # A proxy for time
        self.model_count_history.append(len(self.data))

        # --- Stage 0: Prepare for iteration ---
        num_models_at_start_of_step = len(
            self.data
        )  # Cache initial number for this step
        tree_predictions = [0.0] * num_models_at_start_of_step
        drift_detected_indices = []
        warning_detected_indices = []

        # --- Stage 1: Predictions and Local Updates ---
        for i in range(num_models_at_start_of_step):
            # Make sure lists are long enough; they should be due to _init calls
            # and management in _remove_model/_add_model.
            if (
                i >= len(self.data)
                or i >= len(self._metrics)
                or i >= len(self._dynamic_perf_scores)
                or i >= len(self._accuracy_window)
                or (
                    not self._warning_detection_disabled
                    and i >= len(self._warning_detectors)
                )
                or (
                    not self._drift_detection_disabled
                    and i >= len(self._drift_detectors)
                )
            ):
                # This indicates a bug in list management if it occurs
                logging.error(
                    f"Index {i} out of bounds for core lists in learn_one. Skipping tree."
                )
                continue

            model = self.data[i]
            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree

            # Update standard metric (e.g., MSE, for finding worst model)
            self._metrics[i].update(y_true=y, y_pred=y_pred_tree)

            # Update dynamic performance score (for weighting)
            current_error_abs = abs(y - y_pred_tree)
            error_std_dev = 0.0
            threshold_for_good_pred = float("inf")
            if i < len(self._drift_norm) and self._drift_norm[i].mean.n > 1:
                variance = self._drift_norm[i].get()
                if variance > 0:
                    error_std_dev = math.sqrt(variance)
                    threshold_for_good_pred = (
                        self.dynamic_weighting_error_factor * error_std_dev
                    )
            if current_error_abs <= threshold_for_good_pred:
                self._dynamic_perf_scores[i] *= 0.9
            else:
                self._dynamic_perf_scores[i] *= 1.1
            self._dynamic_perf_scores[i] = max(
                0.01, min(self._dynamic_perf_scores[i], 100.0)
            )

            # Update accuracy window (for pruning)
            # _ensure_accuracy_window_exists(i) # Should be managed by add/remove
            is_accurate_for_pruning = int(
                abs(y - y_pred_tree) <= self.regression_pruning_error_threshold
            )
            self._accuracy_window[i].append(is_accurate_for_pruning)

            # --- Check Detectors ---
            drift_input_for_detectors = None
            if not self._warning_detection_disabled:
                drift_input_for_detectors = self._drift_detector_input(
                    i, y, y_pred_tree
                )
                self._warning_detectors[i].update(drift_input_for_detectors)
                if self._warning_detectors[i].drift_detected:
                    warning_detected_indices.append(i)

            if not self._drift_detection_disabled:
                if (
                    drift_input_for_detectors is None
                ):  # Calculate if not done for warning
                    drift_input_for_detectors = self._drift_detector_input(
                        i, y, y_pred_tree
                    )
                self._drift_detectors[i].update(drift_input_for_detectors)
                if self._drift_detectors[i].drift_detected:
                    drift_detected_indices.append(i)

        # --- Stage 2: Process Detections and Manage Ensemble Size ---
        indices_of_reset_trees = set()

        # Handle warnings
        for i in warning_detected_indices:
            if i >= len(self.data):
                continue  # Tree might have been removed
            if (
                self._background is not None
                and i < len(self._background)
                and self._background[i] is None
            ):
                self._background[i] = self._new_base_model()
                # logging.info(f"🌳 Background learner started for tree {i} due to warning.")
            if i < len(self._warning_detectors):
                self._warning_detectors[i] = self.warning_detector.clone()
            if self._warning_tracker is not None:
                self._warning_tracker[i] += 1

            if i not in self._warned_tree_ids:  # Start monitoring if not already
                self._warned_tree_ids.add(i)
                self._warning_step[i] = current_step
                self._warned_recent_acc[i] = self._get_recent_accuracy(
                    i
                )  # Proxy accuracy
                # logging.info(f"📉 Started monitoring proxy accuracy drop for tree {i} (current acc: {self._warned_recent_acc[i]:.3f}).")

        # Handle drifts (potential model addition or reset)
        # Keep track of indices that were modified to avoid issues if a drift leads to pruning
        # that affects subsequent indices in drift_detected_indices.
        # Iterating over a copy or adjusting indices after removal is safer.
        # For now, let's assume indices in drift_detected_indices are relative to the start of the step.
        # We need to re-evaluate indices if removals happen.

        processed_drift_indices_this_step = set()

        for original_idx in drift_detected_indices:
            # The actual index might have shifted due to prior removals in this loop (if any)
            # This simple loop structure assumes no removals YET, or indices are stable.
            # The _find_worst_model and _remove_model logic later handles index shifts for pruning.
            # If a drift directly causes pruning, careful index management is needed.
            # Here, we add first, then prune later if max_models is exceeded.

            current_idx = (
                original_idx  # This needs careful thought if removals happen mid-loop.
            )
            # For now, assume original_idx is valid against current self.data
            if current_idx in processed_drift_indices_this_step or current_idx >= len(
                self.data
            ):
                continue

            # logging.info(f"💥 Drift detected by detector for tree {current_idx}.")
            if self._drift_tracker is not None:
                self._drift_tracker[current_idx] += 1

            new_tree_candidate = None
            promoted_from_background = False
            if (
                self._background is not None
                and current_idx < len(self._background)
                and self._background[current_idx] is not None
            ):
                new_tree_candidate = self._background[current_idx]
                self._background[current_idx] = None  # Consume background tree
                promoted_from_background = True
                # logging.info(f"➕ Candidate tree from background of tree {current_idx} for potential addition.")

            if new_tree_candidate:  # Try to add this tree
                if len(self.data) >= self.max_models:
                    worst_idx_to_prune = self._find_worst_model()
                    if worst_idx_to_prune is not None:
                        # logging.info(f"📦 Ensemble at max capacity ({self.max_models}). Pruning worst tree {worst_idx_to_prune} before adding.")
                        self._remove_model(worst_idx_to_prune)
                        # If worst_idx_to_prune was current_idx, it's gone.
                        # If worst_idx_to_prune < current_idx, current_idx shifts.
                        # This makes direct addition complex if current_idx itself is pruned.
                        # For simplicity, let's assume current_idx is NOT the one pruned,
                        # or that the pruning logic correctly shifts indices.
                        # The _remove_model adjusts trackers for indices > removed_index.
                        if worst_idx_to_prune < current_idx:
                            current_idx -= (
                                1  # Adjust current_idx if an earlier model was removed
                            )
                    else:  # Cannot prune, so cannot add
                        # logging.warning("Ensemble at max capacity, but no worst model found to prune. Cannot add new tree.")
                        new_tree_candidate = None  # Do not add

                if (
                    new_tree_candidate and len(self.data) < self.max_models
                ):  # Add if there's space or space was made
                    # logging.info(f"➕ Adding new model to ensemble. Ensemble size: {len(self.data)+1}")
                    self.data.append(new_tree_candidate)
                    self._metrics.append(self.metric.clone())
                    if not self._drift_detection_disabled:
                        self._drift_detectors.append(self.drift_detector.clone())
                    if not self._warning_detection_disabled:
                        self._warning_detectors.append(self.warning_detector.clone())
                    if self._background is not None:
                        self._background.append(None)  # Placeholder for new tree
                    self._drift_norm.append(stats.Var())  # For new tree
                    self._accuracy_window.append(
                        collections.deque(maxlen=self.monitor_window)
                    )
                    self._dynamic_perf_scores.append(1.0)  # Default score for new tree
                    # self.n_models should be len(self.data) effectively
                else:
                    # logging.info(f"🔁 Could not add tree from background (max models or no prune). Resetting tree {current_idx} in place.")
                    # Fall through to reset logic if not added
                    new_tree_candidate = None

            if (
                not new_tree_candidate
            ):  # Reset the existing tree at current_idx (if not added from background)
                # logging.info(f"🔁 Resetting tree {current_idx} in place due to drift.")
                self.data[current_idx] = self._new_base_model()
                self._metrics[current_idx] = self.metric.clone()
                if not self._drift_detection_disabled:
                    self._drift_detectors[current_idx] = self.drift_detector.clone()
                if not self._warning_detection_disabled:
                    self._warning_detectors[current_idx] = self.warning_detector.clone()
                self._drift_norm[current_idx] = stats.Var()
                self._accuracy_window[current_idx].clear()
                self._dynamic_perf_scores[current_idx] = 1.0
                indices_of_reset_trees.add(current_idx)
                if current_idx in self._warned_tree_ids:
                    self._clear_warning_state(current_idx)

            # Reset the specific drift detector that triggered for original_idx (now current_idx)
            if not self._drift_detection_disabled and current_idx < len(
                self._drift_detectors
            ):
                self._drift_detectors[current_idx] = self.drift_detector.clone()

            processed_drift_indices_this_step.add(
                original_idx
            )  # Mark original index as processed

        # --- Prune again if ensemble grew beyond max_models (e.g. multiple drifts added trees)
        while len(self.data) > self.max_models:
            worst_idx = self._find_worst_model()
            if worst_idx is not None:
                # logging.info(f"✂️ Ensemble ({len(self.data)}) exceeds max_models ({self.max_models}). Pruning worst tree {worst_idx}.")
                self._remove_model(worst_idx)
            else:
                # logging.warning("Exceeds max_models, but no worst model to prune. Strange state.")
                break  # Avoid infinite loop

        # --- Stage 3: Check Pruning based on Accuracy Drop ---
        self._check_prune_on_accuracy_drop(
            current_step
        )  # This also calls _remove_model

        # --- Stage 4: Update Global State (dynamic weights) and Train ---
        self.n_models = len(self.data)  # Update official count
        self._update_dynamic_weights()  # Recalculate for current ensemble

        # Train all *current* models
        for i in range(len(self.data)):  # Iterate up to current length of self.data
            model_to_train = self.data[i]  # Get current model at index i
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                # Train background model if it exists (check bounds)
                if (
                    not self._warning_detection_disabled
                    and self._background
                    and i < len(self._background)
                    and self._background[i] is not None
                ):
                    self._background[i].learn_one(x=x, y=y, w=k)  # type: ignore
                # Train the main model
                model_to_train.learn_one(x=x, y=y, w=k)
        return self

    def _get_recent_accuracy(self, tree_idx: int) -> float:
        if tree_idx >= len(self._accuracy_window):
            return 0.0  # Should not happen
        acc_deque = self._accuracy_window[tree_idx]
        return (
            sum(acc_deque) / len(acc_deque) if acc_deque else 1.0
        )  # Default to 1.0 if empty (optimistic)

    def _check_prune_on_accuracy_drop(self, current_step: int):
        indices_to_remove = []
        # Iterate over a copy of warned IDs as the set might change during iteration due to _clear_warning_state
        for i in list(self._warned_tree_ids):
            if i >= len(self.data):  # Tree might have been removed by other means
                self._clear_warning_state(i)
                continue

            # If tree was reset due to drift, it should have been cleared from warned_ids.
            # But double check: if it was reset, its accuracy window is new.
            # The age since warning is key.
            age_since_warning = current_step - self._warning_step.get(i, current_step)

            # Stop monitoring if enough time has passed without significant drop
            # or if window is not full yet (no reliable current_acc)
            if (
                age_since_warning > self.monitor_window * 1.5
                or len(self._accuracy_window[i]) < self.monitor_window
            ):
                if (
                    len(self._accuracy_window[i]) == self.monitor_window
                ):  # Only "survive" if monitored for full window
                    # logging.info(f"🌳 Tree {i} survived proxy accuracy drop monitoring.")
                    self._clear_warning_state(i)
                continue  # Keep monitoring if window not full yet, unless very old

            current_proxy_acc = self._get_recent_accuracy(i)
            past_proxy_acc = self._warned_recent_acc.get(
                i, 1.0
            )  # Acc at time of warning

            # Prune if current acc is significantly lower than acc at time of warning
            if (
                past_proxy_acc > 1e-6
                and current_proxy_acc < self.accuracy_drop_threshold * past_proxy_acc
            ):
                # logging.info(f"⚠️ Tree {i} proxy accuracy dropped: {past_proxy_acc:.3f} -> {current_proxy_acc:.3f}. Pruning.")
                indices_to_remove.append(i)
                # _clear_warning_state(i) will be called by _remove_model or after loop
            # else:
            # logging.debug(f"Tree {i} acc: {past_proxy_acc:.3f} -> {current_proxy_acc:.3f}. No prune.")

        # Remove marked trees (in reverse order to maintain index validity)
        for i in sorted(indices_to_remove, reverse=True):
            self._remove_model(i)  # This will also clear warning state for 'i'

    def _find_worst_model(self) -> int | None:
        if not self.data:
            return None

        metric_values = []
        valid_indices = []
        for idx, m_metric in enumerate(self._metrics):
            if idx >= len(self.data):
                continue  # Should not happen
            val = m_metric.get()
            if isinstance(val, (int, float)) and not (
                math.isnan(val) or math.isinf(val)
            ):
                metric_values.append(val)
                valid_indices.append(idx)

        if not metric_values:
            return None

        # self.metric.bigger_is_better is False for typical regression errors (MSE, MAE)
        if (
            self.metric.bigger_is_better
        ):  # We want to remove the one with the smallest good value
            worst_val_idx_in_list = np.argmin(metric_values)
        else:  # We want to remove the one with the largest error value
            worst_val_idx_in_list = np.argmax(metric_values)

        return valid_indices[worst_val_idx_in_list]

    def _remove_model(self, index: int):
        if not (0 <= index < len(self.data)):
            # logging.warning(f"Attempted to remove model at invalid index {index}. Ensemble size: {len(self.data)}")
            return

        # removed_metric_score_val = 'N/A'
        # if index < len(self._metrics): removed_metric_score_val = self._metrics[index].get()
        # logging.info(f"🪓 Removing tree at index {index} (metric: {removed_metric_score_val}). Ensemble size: {len(self.data)-1}")

        del self.data[index]
        if index < len(self._metrics):
            del self._metrics[index]
        if (
            not self._drift_detection_disabled
            and self._drift_detectors
            and index < len(self._drift_detectors)
        ):
            del self._drift_detectors[index]
        if (
            not self._warning_detection_disabled
            and self._warning_detectors
            and index < len(self._warning_detectors)
        ):
            del self._warning_detectors[index]
        if self._background is not None and index < len(self._background):
            del self._background[index]
        if self._drift_norm and index < len(self._drift_norm):  # Manage _drift_norm
            del self._drift_norm[index]
        if self._accuracy_window and index < len(self._accuracy_window):
            del self._accuracy_window[index]
        if self._dynamic_perf_scores and index < len(self._dynamic_perf_scores):
            del self._dynamic_perf_scores[index]
        # Note: _dynamic_weights gets rebuilt by _update_dynamic_weights, so direct removal isn't critical here

        # Adjust pruning tracker indices for elements that came after the removed one
        self._clear_warning_state(index)  # Remove the exact index first
        new_warned_ids = set()
        new_warning_step = {}
        new_warned_recent_acc = {}
        for warned_idx in self._warned_tree_ids:
            if warned_idx > index:  # Shift indices greater than the removed one
                new_idx = warned_idx - 1
                new_warned_ids.add(new_idx)
                if warned_idx in self._warning_step:
                    new_warning_step[new_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_recent_acc:
                    new_warned_recent_acc[new_idx] = self._warned_recent_acc[warned_idx]
            # elif warned_idx < index: # Indices before the removed one are unaffected, keep them (implicitly handled as we build new set)
            #    new_warned_ids.add(warned_idx) # This line is redundant if we iterate self._warned_tree_ids and only modify for > index
            #    if warned_idx in self._warning_step: new_warning_step[warned_idx] = self._warning_step[warned_idx]
            #    if warned_idx in self._warned_recent_acc: new_warned_recent_acc[warned_idx] = self._warned_recent_acc[warned_idx]
            elif warned_idx < index:  # explicit re-add for clarity
                new_warned_ids.add(warned_idx)
                if warned_idx in self._warning_step:
                    new_warning_step[warned_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_recent_acc:
                    new_warned_recent_acc[warned_idx] = self._warned_recent_acc[
                        warned_idx
                    ]

        self._warned_tree_ids = new_warned_ids
        self._warning_step = new_warning_step
        self._warned_recent_acc = new_warned_recent_acc

        # self.n_models = len(self.data) # Updated in learn_one after all modifications

    def _clear_warning_state(self, index: int):
        self._warned_tree_ids.discard(index)
        self._warning_step.pop(index, None)
        self._warned_recent_acc.pop(index, None)

    def plot_model_count(self):
        if not self.model_count_history:
            print("No model count history to plot. Train the model first.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.model_count_history,
            label="Number of Active Models",
            drawstyle="steps-post",
        )
        plt.xlabel("Instances Processed (or learn_one calls)")
        plt.ylabel("Number of Models")
        plt.title(f"{self.__class__.__name__} Ensemble Size Over Time")
        if hasattr(self, "max_models"):
            plt.axhline(
                y=self.max_models,
                color="r",
                linestyle="--",
                label=f"Max Models ({self.max_models})",
            )
        # Min models is usually implicitly 1 if pruning happens, or n_models if no pruning below initial
        # plt.axhline(y=self.n_models, color='g', linestyle=':', label=f'Initial Models ({self.n_models})') # self.n_models changes
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
