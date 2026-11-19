FROM python:3.9-slim

LABEL maintainer="AAVAIL Capstone"
LABEL description="IBM AI Workflow Capstone - AI in Production"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create required directories
RUN mkdir -p logs data model

# Train models at build time
RUN python model/model.py

EXPOSE 8080

ENV FLASK_APP=api/app.py
ENV FLASK_ENV=production

CMD ["python", "api/app.py"]
