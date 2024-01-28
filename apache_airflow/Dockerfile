FROM apache/airflow:2.8.1

USER root
# Install git
RUN apt-get update -y && apt-get install -y git

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get install -y default-jdk && apt-get clean


USER airflow
COPY requirements.txt .
RUN pip install docker==7.0.0

# Avoid cache. This force to reinstall last update of package
RUN echo '2024-01-26 12:00:00' > /tmp/force_cache_invalidator
RUN pip install --upgrade -r requirements.txt