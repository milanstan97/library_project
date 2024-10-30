import click

from utils import (
    load_and_prepare_data,
    train_test_data_split,
    train_and_calibrate_model,
    evaluate_model,
    save_model,
)


@click.command()
@click.option("--input-data", required=True, type=str)
@click.option("--model-name", required=True, type=str)
@click.option("--save-metrics", required=False, type=bool, default=False)
@click.option("--test-ratio", required=False, type=float, default=0.2)
def fit_model(input_data, model_name, save_metrics, test_ratio):

    data = load_and_prepare_data(input_data)
    X_train, X_test, y_train, y_test = train_test_data_split(
        data, test_size=test_ratio
    )

    model = train_and_calibrate_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, save_metrics_to_csv=save_metrics)

    save_model(model, model_name=model_name)


if __name__ == "__main__":
    fit_model()
