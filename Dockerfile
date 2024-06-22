FROM python:3.8.19
LABEL authors="anshujoshi"

# Update and install necessary packages
RUN apt-get update -y && \
    apt-get install -y awscli build-essential libhdf5-dev

# Upgrade pip to the latest version
RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
