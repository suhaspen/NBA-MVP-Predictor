# NBA MVP Predictor - Containerized application for scalable deployment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application (modular backend: api/, services/, data, scripts)
COPY . .

ENV PORT=5000
EXPOSE 5000

# Run the REST API
CMD ["python", "run_app.py"]
