from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from river.tree.nodes.arf_htr_nodes import (
    RandomLeafMean,
    RandomLeafModel,
    RandomLeafAdaptive,
)
from river.tree.splitter import Splitter
from river import base
import copy
import random


class BaseTreeRegressor(HoeffdingTreeRegressor):
    """
    A Hoeffding Tree Regressor base class for use in Adaptive Random Forests (ARFs)
    with randomized leaves.

    Extends River's HoeffdingTreeRegressor to support:
    - Randomized feature selection at each split (controlled by max_features).
    - Customizable leaf prediction strategies: mean, model-based, or adaptive.

    Args:
        max_features (int): Number of features to randomly consider for each split.
        grace_period (int): Number of instances a leaf should observe between split attempts.
        max_depth (int | None): Maximum depth of the tree. None for unlimited depth.
        delta (float): Confidence threshold for the Hoeffding bound.
        tau (float): Tie-breaking threshold for splits.
        leaf_prediction (str): Prediction type at leaf nodes ('mean', 'model', or 'adaptive').
        leaf_model (base.Regressor | None): Optional model used in model or adaptive leaves.
        model_selector_decay (float): Decay rate for model selection in adaptive leaves.
        nominal_attributes (list | None): List of categorical feature names.
        splitter (Splitter | None): Split criterion to use.
        min_samples_split (int): Minimum number of samples required to split a node.
        binary_split (bool): Whether to perform binary splits only.
        max_size (float): Maximum size of the tree in MB.
        memory_estimate_period (int): Frequency of memory usage estimation.
        stop_mem_management (bool): Whether to stop memory management.
        remove_poor_attrs (bool): Whether to remove attributes with poor splits.
        merit_preprune (bool): Whether to prevent splits with insufficient merit.
        rng (random.Random | None): Random number generator for reproducibility.
    """
    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        max_depth: int | None = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            leaf_model=leaf_model,
            model_selector_decay=model_selector_decay,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            min_samples_split=min_samples_split,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.max_features = max_features
        self.rng = rng if rng is not None else random.Random()

    def _new_leaf(self, initial_stats=None, parent=None):
        """
        Create a new leaf node with randomized feature selection.

        Args:
            initial_stats: Initial statistics to use for the leaf.
            parent: Parent node (used to inherit model state if adaptive).

        Returns:
            A RandomLeaf instance depending on the leaf_prediction strategy.
        """
        depth = parent.depth + 1 if parent else 0
        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if (
                parent
                and hasattr(parent, "_leaf_model")
                and parent._leaf_model is not None
            ):
                leaf_model = copy.deepcopy(parent._leaf_model)
            elif self.leaf_model is not None:
                leaf_model = copy.deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return RandomLeafMean(
                initial_stats, depth, self.splitter, self.max_features, self.rng
            )
        elif self.leaf_prediction == self._MODEL:
            return RandomLeafModel(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
                leaf_model=leaf_model,
            )
        else:
            adaptive = RandomLeafAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
                leaf_model=leaf_model,
            )
            if isinstance(parent, RandomLeafAdaptive):
                adaptive._fmse_mean = parent._fmse_mean
                adaptive._fmse_model = parent._fmse_model
            return adaptive
