ARG AIRFLOW_VERSION=3.3.1
FROM apache/airflow:${AIRFLOW_VERSION}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements/mlflow/requirements.txt /tmp/requirements.txt
RUN python -m venv /home/airflow/ml-venv \
    && /home/airflow/ml-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /home/airflow/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/opt/project
WORKDIR /opt/airflow