from pydantic import BaseModel


class PredictionRequest(BaseModel):
    customer_address: str
    customer_age: int
    library_name: str
    book_pages: int
    book_price: float
    book_category: str


class PredictionResponse(BaseModel):
    late_return_prob: float
