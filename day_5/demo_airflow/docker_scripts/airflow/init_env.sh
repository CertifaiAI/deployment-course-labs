export PGPASSWORD=${DB__PASSWORD}
export AIRFLOW__CORE__SQL_ALCHEMY_CONN="postgresql+psycopg2://${DB__USERNAME}:${DB__PASSWORD}@${DB__HOST}:${DB__PORT}/${DB__NAME}"

psql -h ${DB__HOST} -p ${DB__PORT} -U ${DB__USERNAME} ${DB__NAME} -c "CREATE TABLE IF NOT EXISTS airflow__extra_conf(
  conf_name  VARCHAR (255) PRIMARY KEY,
  conf_value VARCHAR (255) NOT NULL
);"

# check on db if admin exists
SECURITY_ALREADY_INITIALIZED=$(cat /opt/airflow/extra/check_init.sql | psql -h ${DB__HOST} -p ${DB__PORT} -U ${DB__USERNAME} ${DB__NAME} -t | xargs | head -c 1)

# Initialize db
airflow db upgrade

if [ "${SECURITY_ALREADY_INITIALIZED}" == "0" ]; then
  echo "Creating admin user.."
	airflow users create -r Admin -u "$SECURITY__ADMIN_USERNAME" -e "$SECURITY__ADMIN_EMAIL" -f "$SECURITY__ADMIN_FIRSTNAME" -l "$SECURITY__ADMIN_LASTNAME" -p "$SECURITY__ADMIN_PASSWORD"
	cat /opt/airflow/extra/set_init.sql | psql -h ${DB__HOST} -p ${DB__PORT} -U ${DB__USERNAME} ${DB__NAME} -t
fi

export PYTHONPATH=/opt/airflow/:$PYTHONPATH

pip install --no-cache-dir -r /requirements.txt

curl -o /opt/airflow/scripts/dog_classifier.pkl https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/dog_classifier.pkl