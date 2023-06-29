FROM python:3.9-slim

RUN apt-get update
RUN apt-get install libgomp1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY deployment/model_predictor/requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./src /sample_solution/src
COPY ./service.py /sample_solution/service.py
COPY ./features.py /sample_solution/features.py
COPY ./configuration.yaml /sample_solution/configuration.yaml
# COPY predict.sh /sample_solution/predict.sh
WORKDIR /sample_solution

