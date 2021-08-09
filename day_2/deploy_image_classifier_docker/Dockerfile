FROM python:3.8.5-slim

WORKDIR /image_classifier

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY ./api_server ./api_server

ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR"

CMD ["python", "./api_server/main.py"]