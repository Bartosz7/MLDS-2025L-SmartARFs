import matplotlib.pyplot as plt
import logging
import time
from river import metrics


def evaluate_model(
    model,
    stream,
    name,
    print_interval=10000,
    metric_class=metrics.MAE,
    track_model_count=False,
    verbose=True
):
    """
    Evaluate a River/River-extended regressor model on a data stream.

    Args:
        model: River-compatible regressor with `learn_one`, `predict_one`, and optionally `__len__` and `n_drifts_detected`.
        stream: Iterable yielding (x, y) pairs.
        name (str): Display name of the model.
        print_interval (int): Instance step interval to log progress.
        metric_class: River metric class, e.g., metrics.MAE.
        track_model_count (bool): Track number of models in the ensemble.
        verbose (bool): If True, logs progress info using the logging module.

    Returns:
        dict: {
            "name": str,
            "metric": river.Metric,
            "mae_history": list,
            "model_count_history": list or None,
            "duration": float (seconds)
        }
    """
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)

    metric_eval = metric_class()
    mae_history = []
    model_count_history = [] if track_model_count else None

    if verbose:
        logger.info(f"\nEvaluating {name}...")

    start_time = time.time()
    for i, (x, y) in enumerate(stream):
        y_pred = model.predict_one(x)
        if y_pred is not None:
            metric_eval.update(y, y_pred)
            mae_history.append(metric_eval.get())
        else:
            mae_history.append(mae_history[-1] if mae_history else 0.0)

        model.learn_one(x, y)

        if track_model_count and hasattr(model, '__len__'):
            model_count_history.append(len(model))

        if verbose and (i + 1) % print_interval == 0:
            n_models = len(model) if hasattr(model, '__len__') else '?'
            n_drifts = model.n_drifts_detected() if hasattr(model, 'n_drifts_detected') else '?'
            logger.info(f"  {name} - Step {i+1}, MAE: {metric_eval.get():.4f}, "
                        f"Num Models: {n_models}, Drifts: {n_drifts}")

    duration = time.time() - start_time
    if verbose:
        logger.info(f"{name} evaluation time: {duration:.2f} seconds")

    return {
        "name": name,
        "metric": metric_eval,
        "mae_history": mae_history,
        "model_count_history": model_count_history,
        "duration": duration
    }
