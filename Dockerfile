FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train at build time if models are missing (needs data/*.csv in the image)
RUN [ -f models/recommender.joblib ] || python train.py
ENV PORT=8000
EXPOSE 8000
CMD gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
