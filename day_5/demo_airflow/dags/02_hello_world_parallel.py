from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Following are defaults which can be overridden later on
default_args = {
    'start_date': datetime(2016, 4, 15),
    'retry_delay': timedelta(minutes=1),
}

with DAG('helloworld_parallel', default_args=default_args) as dag:

    # t1, t2, t3 and t4 are examples of tasks created using operators

    t1 = BashOperator(
        task_id='task_1',
        bash_command='echo "Hello World from Task 1"',
        dag=dag)

    t2 = BashOperator(
        task_id='task_2',
        bash_command='echo "Hello World from Task 2"',
        dag=dag)

    t3 = BashOperator(
        task_id='task_3',
        bash_command='echo "Hello World from Task 3"',
        dag=dag)

    t4 = BashOperator(
        task_id='task_4',
        bash_command='echo "Hello World from Task 4"',
        dag=dag)
    
    t1 >> [t2, t3] >> t4