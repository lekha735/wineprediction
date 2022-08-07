FROM reddotpay/docker-pyspark

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "prediction.py"]
