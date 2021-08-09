FROM python:3.8.5-slim

WORKDIR /app

RUN apt-get update
RUN apt install -y python3-opencv

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./app

COPY ./sample_data ./sample_data

ENV PYTHONPATH="$PYTHONPATH:$WORKDIR"

CMD ["python", "./app/main.py"]

