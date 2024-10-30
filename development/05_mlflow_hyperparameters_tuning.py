import logging
import os
import time
import warnings
from functools import partial

import mlflow
import pandas as pd
from hyperopt import fmin, tpe, Trials, hp
from loguru import logger
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.getLogger("mlflow").setLevel(logging.ERROR)


warnings.filterwarnings("ignore")

dir = os.path.dirname(os.path.realpath("__file__"))
root = os.path.realpath(os.path.join(dir, ".."))
merged_data_path = os.path.join(root, "data", "merged_data")

USE_COLS = [
    "customer_library_distance_km",
    "book_pages",
    "book_price",
    "customer_age",
    "book_categories",
    "returned_late",
]


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution time: {round(end_time-start_time, 2)} s")
        return result

    return wrapper


def get_model_metrics(y_true, y_prob, prefix):

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
        f"{prefix}_ball_acc": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    return metrics


@timer
def objective_function(params, X_train, y_train, experiment_id):

    params.update({"n_estimators": int(params["n_estimators"])})
    params.update({"max_depth": int(params["max_depth"])})
    params.update({"learning_rate": float(params["learning_rate"])})

    logger.info(
        f"n_estimators={params['n_estimators']}, max_depth="
        f"{params['max_depth']}, learning_rate={params['learning_rate']}"
    )

    scale_pos_weight = round(
        y_train.value_counts()[0] / y_train.value_counts()[1], 2
    )

    model = XGBClassifier(
        random_state=42, scale_pos_weight=scale_pos_weight, **params
    )

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:

        roc_auc = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc"
        ).mean()

        # explicitly fit the model on the entire training data
        model.fit(X_train, y_train)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("cv_roc_auc", roc_auc)
        mlflow.xgboost.log_model(model, f"{run.info.run_id}-model")

    return -roc_auc  # min(-f) = max(f)


if __name__ == "__main__":

    data = pd.read_csv(
        os.path.join(merged_data_path, "modeling_data.csv"), usecols=USE_COLS
    )
    data["book_category_medicine"] = data["book_categories"].apply(
        lambda x: 1 if x == "Medicine" else 0
    )

    data.drop(columns=["book_categories"], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["returned_late"]),
        data["returned_late"],
        test_size=0.2,
        stratify=data["returned_late"],
        random_state=42,
    )

    search_space = {
        "n_estimators": hp.quniform("n_estimators", 15, 25, 1),
        "max_depth": hp.quniform("max_depth", 2, 10, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    }

    try:
        experiment_id = mlflow.create_experiment(
            name="mlflow_experiment",
            artifact_location="mlflow_artifacts",
            tags={
                "mlflow.note.content": "XGBoost hyperparameters tuning using"
                "hyperopt"
            },
        )
    except:
        logger.info(f"Experiment already exists")
        experiment_id = mlflow.get_experiment_by_name(
            "mlflow_experiment"
        ).experiment_id

    with mlflow.start_run(
        run_name="hyperparameter_tuning", experiment_id=experiment_id
    ) as run:

        best_params = fmin(
            fn=partial(
                objective_function,
                X_train=X_train,
                y_train=y_train,
                experiment_id=experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,  # Bayesian optimization
            max_evals=20,
            trials=Trials(),
        )

        # training final model with best hyperparameters
        best_params.update({"n_estimators": int(best_params["n_estimators"])})
        best_params.update({"max_depth": int(best_params["max_depth"])})
        best_params.update(
            {"learning_rate": float(best_params["learning_rate"])}
        )

        scale_pos_weight = round(
            y_train.value_counts()[0] / y_train.value_counts()[1], 2
        )

        model = XGBClassifier(
            random_state=42, scale_pos_weight=scale_pos_weight, **best_params
        )

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = get_model_metrics(y_test, y_prob, prefix="best_model_test")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        model_signature = infer_signature(X_train, y_train)
        mlflow.xgboost.log_model(
            model, "best_model", signature=model_signature
        )
