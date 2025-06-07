"""
This module is to quickly instantiate new models
"""
from river.forest import ARFRegressor
from river.drift import ADWIN
from river import metrics
from .smart_arf_regressor import SmartARFRegressor


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
