#FROM ubuntu:16.04
FROM python:3.5-slim

MAINTAINER Nitin Kaushik

RUN apt-get update -y && apt-get install -y python-pip python-dev

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
