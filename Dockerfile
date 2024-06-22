FROM python:3.8
RUN apt-get update -y && apt-get install awscli -y
RUN pip install --upgrade pip
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
