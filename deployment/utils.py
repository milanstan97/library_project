import os
import pickle
import time
from datetime import date

import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

wd = os.path.dirname(os.path.realpath(__file__))
root = os.path.realpath(os.path.join(wd, ".."))
USE_COLS = [
    "customer_library_distance_km",
    "book_pages",
    "book_price",
    "customer_age",
    "book_categories",
    "returned_late",
]


def load_and_prepare_data(data_file: str) -> pd.DataFrame:
    """
    Load and preprocess data for model training.

    Args:
        data_file (str): The name of the CSV file (without extension)
        containing the modeling data.

    Returns:
        pd.DataFrame: Processed data with relevant columns and a new
        'book_category_medicine' feature.
    """
    data_path = os.path.join(root, "data", "merged_data")
    data = pd.read_csv(
        os.path.join(data_path, f"{data_file}.csv"), usecols=USE_COLS
    )

    data["book_category_medicine"] = data["book_categories"].apply(
        lambda x: 1 if x == "Medicine" else 0
    )

    return data.drop(columns=["book_categories"])


def train_test_data_split(
    data: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        data (pd.DataFrame): The dataset with features and labels.
        test_size (float): Proportion of the dataset to include in the test
        split.

    Returns:
        tuple: X_train, X_test (pd.DataFrame) - Feature sets for training and
               testing.
               y_train, y_test (pd.Series) - Labels for training and testing.
    """
    X = data.drop(columns=["returned_late"])
    y = data["returned_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(f"Features: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test


def train_and_calibrate_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> CalibratedClassifierCV:
    """
    Train XGBoost classifier and calibrate it using sigmoid method.

    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.

    Returns:
        CalibratedClassifierCV: Calibrated XGBoost classifier.
    """
    scale_pos_weight = round(
        y_train.value_counts()[0] / y_train.value_counts()[1], 2
    )

    start_time = time.time()
    model = XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(model, method="sigmoid")
    calibrated_model.fit(X_train, y_train)
    logger.info(
        f"Fitting and calibrating model completed in"
        f" {round(time.time()-start_time,2)} seconds"
    )

    return calibrated_model


def evaluate_model(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_metrics_to_csv: bool,
) -> None:
    """
    Evaluate the model on the test set using ROC AUC, balanced accuracy and
    f1 weighted metrics. Optionally save metrics to a CSV file.

    Args:
        model (CalibratedClassifierCV): Trained and calibrated model.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True labels for the test set.
        save_metrics_to_csv (bool): Whether to save metrics to a CSV file
        (for model monitoring).

    Returns:
        None
    """
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    metrics = {
        "roc_auc": [round(roc_auc_score(y_test, y_prob_test), 2)],
        "ball_acc": round(balanced_accuracy_score(y_test, y_pred_test), 2),
        "f1_weighted": round(
            f1_score(y_test, y_pred_test, average="weighted"), 2
        ),
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df["date"] = date.today().strftime("%Y-%m-%d")
    metrics_df = metrics_df[["date", "roc_auc", "ball_acc", "f1_weighted"]]

    logger.info(f"Model metrics: \n {metrics_df}")

    if save_metrics_to_csv:
        metrics_df.to_csv(os.path.join(wd, "model_metrics.csv"), index=False)


def save_model(model: CalibratedClassifierCV, model_name: str) -> None:
    """
    Save the trained model to a pickle file.

    Args:
        model (CalibratedClassifierCV): Trained and calibrated model.
        model_name (str): The name of the model.

    Returns:
        None
    """
    model_path = os.path.join(wd, f"{model_name}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model {model_name} successfully saved")
