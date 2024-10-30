FROM python:3.9-slim

WORKDIR /app

COPY ./library_model_v1.pkl ./requirements.txt /app/
COPY ./api /app/api

RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]