FROM python:3.9.18-slim-bullseye

WORKDIR /mlflow/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow ui --host 0.0.0.0 --port 5000