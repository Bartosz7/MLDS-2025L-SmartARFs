"""
This module is to quickly instantiate new models
"""
from river.forest import ARFRegressor
from river.drift import ADWIN
from river import metrics
from .smart_arf_regressor import SmartARFRegressor
from .arf_regressor_dw import ARFRegressorDynamicWeights
from .smart_arf_dw_regressor import SmartARFDynamicWeightsRegressor


def make_standard_arf(n_models=10,
                      seed=42,
                      lambda_value=6,
                      grace_period=50,
                      leaf_prediction="adaptive",
                      metric=metrics.MAE(),
                      drift_detector=ADWIN(delta=0.001),
                      warning_detector=ADWIN(delta=0.01)):
    return ARFRegressor(
        n_models=n_models,
        seed=seed,
        lambda_value=lambda_value,
        grace_period=grace_period,
        leaf_prediction=leaf_prediction,
        metric=metric,
        drift_detector=drift_detector,
        warning_detector=warning_detector,
    )


def make_arf_dw(n_models=10,
                seed=42,
                lambda_value=6,
                grace_period=50,
                leaf_prediction="adaptive",
                metric=metrics.MAE(),
                drift_detector=ADWIN(delta=0.001),
                warning_detector=ADWIN(delta=0.01),
                error_mode="std",
                error_threshold_factor=1.0):
    return ARFRegressorDynamicWeights(
        n_models=n_models,
        seed=seed,
        lambda_value=lambda_value,
        grace_period=grace_period,
        leaf_prediction=leaf_prediction,
        metric=metric,
        drift_detector=drift_detector,
        warning_detector=warning_detector,
        error_mode=error_mode,
        error_threshold_factor=error_threshold_factor,
    )


def make_smart_arf(n_models=10,
                   max_models=20,
                   min_ensemble_size=5,
                   seed=42,
                   lambda_value=6,
                   grace_period=6,
                   leaf_prediction="adaptive",
                   metric=metrics.MAE(),
                   drift_detector=ADWIN(delta=0.01),
                   warning_detector=ADWIN(delta=0.1),
                   verbose=True):

    return SmartARFRegressor(
        n_models=n_models,
        max_models=max_models,
        min_ensemble_size=min_ensemble_size,
        seed=seed,
        lambda_value=lambda_value,
        grace_period=grace_period,
        leaf_prediction=leaf_prediction,
        metric=metric,
        drift_detector=drift_detector,
        warning_detector=warning_detector,
        verbose_logging=verbose,
    )


def make_smart_arf_dw(n_models=10,
                      max_models=20,
                      min_ensemble_size=5,
                      seed=42,
                      lambda_value=6,
                      grace_period=6,
                      leaf_prediction="adaptive",
                      metric=metrics.MAE(),
                      drift_detector=ADWIN(delta=0.01),
                      warning_detector=ADWIN(delta=0.1),
                      regression_pruning_error_threshold=0.1,
                      accuracy_drop_threshold=0.5,
                      monitor_window=100,
                      error_mode="std",
                      error_threshold_factor=1.0,
                      verbose=True):
    return SmartARFDynamicWeightsRegressor(
        n_models=n_models,
        max_models=max_models,
        min_ensemble_size=min_ensemble_size,
        seed=seed,
        lambda_value=lambda_value,
        grace_period=grace_period,
        leaf_prediction=leaf_prediction,
        metric=metric,
        drift_detector=drift_detector,
        warning_detector=warning_detector,
        regression_pruning_error_threshold=regression_pruning_error_threshold,
        accuracy_drop_threshold=accuracy_drop_threshold,
        monitor_window=monitor_window,
        error_mode=error_mode,
        error_threshold_factor=error_threshold_factor,
        verbose_logging=verbose,
    )
