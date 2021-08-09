# Demo for Apache Airflow
In this repo, we will be showing you how you can ochestrate pipelines using Apache Airflow.

- [Demo for Apache Airflow](#demo-for-apache-airflow)
  - [1.0 Introduction to Apache Airflow](#10-introduction-to-apache-airflow)
  - [2.0 Quick Start](#20-quick-start)
    - [2.1 Step 1: Build the docker image](#21-step-1-build-the-docker-image)
    - [2.2  Step 2: Initialize database](#22--step-2-initialize-database)
    - [2.3  Step 3: It's Airflow time!](#23--step-3-its-airflow-time)
      - [2.3.1 Run an example](#231-run-an-example)
      - [2.3.2 Unpause and trigger the dag](#232-unpause-and-trigger-the-dag)
      - [2.3.3 Success](#233-success)
    - [2.4 To quit](#24-to-quit)
  - [DAGs Development](#dags-development)
  - [Tested on](#tested-on)
  - [References](#references)

## 1.0 Introduction to Apache Airflow
In short, Airflow is an open source tool to programmatically author, schedule and monitor worflows. It is build purely using Python and easily scalable. You may refer to the official website [here](https://airflow.apache.org/) to learn more about it.

## 2.0 Quick Start 
Airflow can be installed directly on to the local machine environment. It is also distributed as Docker Image. The official instructions can be found [here](https://airflow.apache.org/docs/apache-airflow/stable/installation.html). 

In this demo, we will be using the official docker image but will also build a thin layer on top of it to suit our use cases. Without further ado, let's get started.

### 2.1 Step 1: Build the docker image
The first step is to build a docker image based on the docker-compose yaml files. Before you start, you need to make sure that you have [installed docker](https://docs.docker.com/get-docker/).

Then, go to the `demo-airflow` project folder and
open a terminal. Run the following command in the terminal:
```console
docker-compose up --build
```
**Note**:

If you're using windows to build the docker image, you may encounter the error
`/usr/bin/env: ‘bash\r’: No such file or directory`
To solve this, you need to re-clone the repo with additional `--config core.autocrlf=input` argument.
```console
git clone https://github.com/CertifaiAI/deployment-course-labs.git --config core.autocrlf=input
```

![docker-compose-up-gif](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/docker_compose_up.gif)

Once you see two `AIRFLOW` appear at the terminal. Proceed to the next step.

![two-airflows](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/two_airflows.png)

### 2.2  Step 2: Initialize database

Open a new terminal and run the following line in the terminal.
```console
docker exec -it demo_airflow_server_1 bash ./init_env.sh
```
Wait until the whole process complete.

### 2.3  Step 3: It's Airflow time!
It is time for us to launch the airflow. You can go to your favourite web browser and access it via
http://localhost:8080/

If you're prompted to login, kindly key in the following username and password:

username: `admin`
password: `admin`

![localhost_airflow](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/localhost_airflow.png)

#### 2.3.1 Run an example
To run one of the example, simply click one of the dags.
![run_example_step_1](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/run_example_step_1.png)

#### 2.3.2 Unpause and trigger the dag
Click buttons at 2 and 3. In step 3, choose the option 'Trigger DAG'

![run_example_step_2_3](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/run_example_step_2_3.png)

#### 2.3.3 Success
If the dashboard shows green, it means that the dag run successfully.

![run_example_step_4](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_5/demo_airflow/run_example_step_4.png)

### 2.4 To quit

Press `Ctrl+c` on the terminal running the Airflow webserver.

If you want to stop the container, run the following code:
```console
docker-compose down
```
## DAGs Development
The `dags` in `demo_airflow` folder is directly mounted to the respective `dags` folder in `\opt\airflow\dags\` of the container.

You may edit / add extra dags to the host `dags` folder and it will reflect the changes to the `dags` in container as it has been mounted.

## Tested on
- Ubuntu 18.04
- Python 3.6
  
## References
- https://towardsdatascience.com/setting-up-apache-airflow-2-with-docker-e5b974b37728
- https://towardsdatascience.com/getting-started-with-airflow-using-docker-cd8b44dbff98
- https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html
- https://airflow.apache.org/docs/apache-airflow/stable/start/local.html
