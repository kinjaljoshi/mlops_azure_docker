FROM python:3.8-slim

# Set environment variable to prevent frontend from trying to access debconf
ENV DEBIAN_FRONTEND=noninteractive

# Install Java runtime for H2O
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends default-jre && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt	

CMD ["python","./app/inference_pipeline.py"]