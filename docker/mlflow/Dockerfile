FROM python:3.12-slim

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        "mlflow==2.22.0" \
        "SQLAlchemy>=2.0,<2.1" \
        "pandas<3" \
        "numpy<3" \
        "psycopg2-binary==2.9.10" \
    && groupadd --gid ${APP_GID} mlflow \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R ${APP_UID}:${APP_GID} /mlflow

USER mlflow
WORKDIR /mlflow