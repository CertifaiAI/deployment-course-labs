FROM apache/airflow:2.1.2-python3.8
USER root
# INSTALL TOOLS
RUN apt-get update \
&& apt-get -y install libaio-dev \
&& sudo apt-get install libgomp1 \
&& apt-get install postgresql-client

USER airflow

COPY requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt

# COPY SQL SCRIPT
COPY docker_scripts/airflow/check_init.sql ./extra/check_init.sql
COPY docker_scripts/airflow/set_init.sql ./extra/set_init.sql

# ENTRYPOINT SCRIPT
COPY docker_scripts/airflow/init.sh ./init.sh
COPY docker_scripts/airflow/init_env.sh ./init_env.sh

ENV PYTHONPATH "$PYTHONPATH:/opt/airflow/"

ENTRYPOINT ["./init.sh"]