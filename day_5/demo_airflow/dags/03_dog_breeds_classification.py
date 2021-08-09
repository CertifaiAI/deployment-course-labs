import json
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from scripts.classifier import get_classifier

default_args = {
    'start_date': datetime(2021, 1, 1)
}

def _check_json_stat():
    with open('/tmp/dog.json') as f:
        dog =  json.load(f)
        status = dog['status']
        if status != 'success':
            raise ValueError(f'API status: {status}')
        else:
            return dog

def _get_img_url(ti):
    data = ti.xcom_pull(task_ids=['check_json_stat'])[0]
    print(f'the data is {data}')
    data_url = data['message']
    return data_url

def _get_img_lbl(ti):
    data = ti.xcom_pull(task_ids=['check_json_stat'])[0]
    data_url = data['message']
    breed_name = data_url.split('/')[-2]
    return breed_name

def _predict_img():
    learner = get_classifier()
    results = learner.predict("/tmp/dog.jpg")
    pred_dog_breed = results[0]
    pred_dog_breed = pred_dog_breed.split('-')[1]
    return pred_dog_breed

with DAG('dog_breeds_classification', schedule_interval='*/1 * * * *', default_args=default_args, catchup=False) as dag:
    fetch_json = BashOperator(
        task_id = "fetch_json",
        bash_command = "curl -o /tmp/dog.json -L 'https://dog.ceo/api/breeds/image/random'"
    )

    check_json_stat = PythonOperator(
        task_id = "check_json_stat",
        python_callable = _check_json_stat
    )

    get_img_url = PythonOperator(
        task_id = "get_img_url",
        python_callable = _get_img_url
    )

    get_img_lbl = PythonOperator(
        task_id = "get_img_lbl",
        python_callable = _get_img_lbl
    )

    download_img = BashOperator(
        task_id = "download_img",
        bash_command = 'echo download data from "{{ti.xcom_pull(task_ids="get_img_url")}}" && \
            curl -o /tmp/dog.jpg {{ti.xcom_pull(task_ids="get_img_url")}}'
    )

    predict_img = PythonOperator(
        task_id = "predict_img",
        python_callable = _predict_img
    )

    compare_result = BashOperator(
        task_id = "compare_result",
        bash_command = 'echo predicted class: {{ti.xcom_pull(task_ids="predict_img")}} actual_class: {{ti.xcom_pull(task_ids="get_img_lbl")}}'
    )

    fetch_json >> check_json_stat >> get_img_url >> download_img >> predict_img >> compare_result
    fetch_json >> check_json_stat >> get_img_lbl >> compare_result

