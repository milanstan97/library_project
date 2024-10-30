# Library ML Model

## Development Setup

First, create and activate a Python virtual environment. Execute the following commands in the project root:

```shell
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate for Windows
pip install -r requirements-dev.txt
```

Navigate to the development directory and run each notebook in the following order:

1.  `01_data_cleaning.ipynb`
2.  `02_merge_data_and_create_features.ipynb`
3.  `03_eda.ipynb`
4.  `04_models.ipynb`

If you want to perform hyperparameter tuning with MLflow, run the following commands:

```shell
cd development
python 05_mlflow_hyperparameters_tuning.py
mlflow ui
```

## Deployment Setup

Run the following commands to train the production model and build the model Docker image. Make sure to activate the virtual environment before running the commands:

```shell
cd deployment
python ./fit_model.py --input-data "modeling_data" --model-name "library_model_v1"
docker build -t library-late-return .
docker run -d -p 8000:8000 library-late-return
```

Alternatively, you can pull the Docker image from DockerHub and run it:

```shell
docker pull milan997/library-model:1.0
docker run -d -p 8000:8000 milan997/library-model:1.0
```

The model is now accessible at [http://127.0.0.1:8000/api/v1/predict/](http://127.0.0.1:8000/api/v1/predict/). Below is an example of the data payload:

```json
{
 "customer_address": "10962 N Swift Ct, Portland, OR 97203",
 "customer_age": 30,
 "library_name": "Multnomah County Library Northwest",
 "book_pages": 629,
 "book_price": 197.99,
 "book_category": "Business & Economics"
}
```

For any questions, feel free to contact Milan at milan.s997@gmail.com.
