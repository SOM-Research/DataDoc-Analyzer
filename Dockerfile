FROM python:3.10.11-bullseye

RUN apt-get update
RUN apt-get -y install git

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./src ./src
COPY ./api.py .
COPY ./app.py .
COPY ./README.md .
COPY ./vectors .
COPY ./gradio_cached_examples .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
