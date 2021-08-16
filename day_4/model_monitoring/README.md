# Model Monitoring with Prometheus and Grafana

This repository will demonstrate on some basic model monitoring metrics using production grade tools such as Prometheus and Grafana. The model used for this demonstration is similar to the fruit classifier model used in Day 2. Locust will be used to simulate traffic into the model server.

![Sample Dashboard Image](https://user-images.githubusercontent.com/76937732/125888089-313ac0a7-5707-4095-902d-f987998b0400.png)

---

## Table of Contents
1. [Components](#components)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Deeper Insights](#deeper-insights)
   - [Instrumentation](#instrumentation)
   - [Prometheus](#prometheus)
   - [Locust](#locust)
   - [Grafana](#grafana)
6. [References](#references)

## Components
- ML model served via `FastAPI`
- Server metrics exported via `starlette-exporter`
- CPU metrics exported via `node-exporter`
- Simulate production traffic via `locust`
- Scrape and store metrics via `Prometheus`
- Visualize collected metrics via `Grafana`

## System Requirements

The guide and code for this repository was written and tested on: 

* Ubuntu 20.04 LTS
* Python version 3.8.5 

## Architecture

<center>

![Model Monitoring Container Architecture](https://user-images.githubusercontent.com/76937732/126062751-ab043f85-da0c-4e33-a48d-c5043aecb774.png)

</center>

## Quick Start

1. Make sure that your working directory is in `model_monitoring`
2. You need to have `fruit_classifier_state_dict.pt` in `./app/artifacts`. You may directly download the file from this [link](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt) and place it in the target directory.

    Alternatively, you can just run the `download_models.py` script. The script will download the weights file and save it into the folder.

    ```sh
    python3 download_models.py
    ```
3. Run the following command to spin up the monitoring stack

    ```sh
    docker-compose up -d
    ``` 
4. Go to `http://localhost:3000`. Login with `admin` as username and password. Click `Skip`.
5. Select 'Add your first data source' and choose 'Prometheus'
6. At the URL field, input `http://prometheus:9090` and at the scrape interval, input 1s. Hit 'Save & test' at the bottom.
7. Hover your cursor over the plus sign at the navigation bar on your left, select 'Import'
8. Choose 'Upload JSON file' and select `./dashboard/template_linux.json` located in this directory.
9. At the 'Prometheus' field, choose 'Prometheus (default)' at the dropdown and hit 'Import'. You should now be able to see the dashboard.
10. Next, to simulate traffic, create a virtual environment and install the loading dependencies. Run the following line by line. 

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r load_test/requirements.txt
    ```
    If you are on a Windows machine with Anaconda/ Miniconda installed, you may use the following code snippet to create an environment called `locustwin` and install the necessary dependencies.

    ```
    conda create -n locustwin
    conda activate locustwin
    pip install -r load_test\requirements_win.txt
    ```
11. Initiate simulated production traffic with the following command. You may use `CTRL+C` to stop it mid-way.

    ```
    locust --config load_test/locust.conf
    ```
12. Run the following command to bring down the monitoring stack once you are done.

    ```
    docker-compose down
    ```

## Deeper Insights
---
### **Instrumentation**
The `model_server` was instrumented with the use of a custom library called `starlette-exporter`. By default, this library offers the following metrics:-

|   |           Name            |    Type   |            Description            |
|---|:-------------------------:|:---------:|:---------------------------------:|
| 1 | requests                  | Counter   | Total HTTP requests               |
| 2 | requests_duration_seconds | Histogram | HTTP request duration, in seconds |
| 3 | request_in_progress       | Gauge     | Total HTTP requests in progress   |

There was one additional metric custom-made for this machine learning model monitoring use case in `monitoring.py`.

|   |     Name    |   Type  |          Description         |
|---|:-----------:|:-------:|:----------------------------:|
| 1 | predictions | Counter | Total of all predicted class |

### **Prometheus**
Prometheus was configured to scrape three endpoints at a scrape interval of 1s.

1. Prometheus (itself)
2. Node
3. Model server

### **Locust**
By default, it spawns 100 users and runs for 5 mins. 

### **Grafana**
Version: 8.0.5

## References:
- [A simple solution for monitoring ML systems](https://www.jeremyjordan.me/ml-monitoring/)
- [ml-monitoring](https://github.com/jeremyjordan/ml-monitoring)