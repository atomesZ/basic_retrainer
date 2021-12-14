FROM python:3.8.10-slim-buster

WORKDIR /app

RUN mkdir models

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir

COPY main.py main.py

CMD ["python3", "main.py"]
