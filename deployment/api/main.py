import warnings

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from api.utils import load_model, load_library_map, calculate_distance

warnings.filterwarnings("ignore")
import numpy as np

library_map = load_library_map()
model = load_model("library_model_v1")
app = FastAPI(
    title="Library model API",
    description="API for predicting late book returns",
    version="1.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Library model API"}


@app.post("/api/v1/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    customer_address = request.customer_address
    customer_age = request.customer_age
    library_name = request.library_name
    book_pages = request.book_pages
    book_price = request.book_price
    book_category = request.book_category

    library_address = library_map.get(library_name, "")
    customer_library_distance_km = calculate_distance(
        library_address, customer_address
    )
    if np.isnan(customer_library_distance_km):
        raise HTTPException(
          status_code=400,
          detail="Please check if you entered the customer address correctly!",
        )

    book_category_medicine = 1 if book_category.lower() == "medicine" else 0

    model_input = np.array(
        [
            customer_age,
            customer_library_distance_km,
            book_price,
            book_pages,
            book_category_medicine,
        ]
    ).reshape(1, -1)
    late_return_prob = model.predict_proba(model_input)[0][1]

    return {"late_return_prob": round(late_return_prob, 2)}
