from datetime import datetime
import json
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

default_args = {
    'start_date': datetime(2021, 1, 1)
}

def preprocess_json() -> dict:
    with open('/tmp/covid.json') as f:
        covid_data =  json.load(f)
        data = covid_data['data']
        if len(data) == 0:
            raise ValueError("No data is found, either the api is down or check your date.")
        return data

def _filter_data(ti) -> dict:
    data = ti.xcom_pull(task_ids=['preprocess_json'])[0]
    print(data)
    filtered_data =  {
        'last_update': data['last_update'],
        'active': data['active'],
        'confirmed': data['confirmed'],
        'death': data['deaths'],
        'recovered': data['recovered']
    }
    return filtered_data

def save_to_json(ti):
    filtered_data = ti.xcom_pull(task_ids=['filter_data'])[0]
    with open('/tmp/covid_filtered.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f,  ensure_ascii=False, indent=4)

with DAG('covid_processing', schedule_interval='@daily', default_args=default_args, catchup=False) as dag:
    
    #you may replace {{ds}} to the date to get dynamic date updates
    get_json_data = BashOperator(
        task_id = "get_json_data",
        bash_command = "curl --request GET \
	--url 'https://covid-19-statistics.p.rapidapi.com/reports/total?date=2021-07-14' \
	--header 'x-rapidapi-host: covid-19-statistics.p.rapidapi.com' \
	--header 'x-rapidapi-key: 479d9afa53mshc4d5d2a8db25d0cp184771jsn91e42e5e8869' \
    --output '/tmp/covid.json'"
    )

    preprocess_json = PythonOperator(
        task_id = 'preprocess_json',
        python_callable = preprocess_json,
        do_xcom_push = True
    )

    filter_data = PythonOperator(
        task_id = 'filter_data',
        python_callable = _filter_data
    )

    save_to_json = PythonOperator(
        task_id = 'save_to_json',
        python_callable = save_to_json
    )

    notification = BashOperator(
        task_id = 'notification',
        bash_command = 'cat /tmp/covid_filtered.json'
    )

    get_json_data >> preprocess_json >> filter_data >> save_to_json >> notification
