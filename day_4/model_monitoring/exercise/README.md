# Exercise
This is an exercise to consolidate your understanding in using the model monitoring tools shared today: Prometheus, Grafana and Locust.

You will need to create four containers.

1. `model_server`: Using any existing model servers from the Lab Exercises in Day 2, instrument that server to expose the following metrics:-

    |   |           Name            |    Type   |            Description            |
    |---|:-------------------------:|:---------:|:---------------------------------:|
    | 1 | requests                  | Counter   | Total HTTP requests               |
    | 2 | requests_duration_seconds | Histogram | HTTP request duration, in seconds |
    | 3 | request_in_progress       | Gauge     | Total HTTP requests in progress   |
    | 4 | predictions               | Counter   | Total of all predicted class      |

    You may stick to using only the native client libraries or utilise custom instrumentator libraries such as `starlette-exporter` and `prometheus-fastapi-instrumentator`.

2. `node`: Setup `node-exporter` to export system metrics.
3. `prometheus`: Setup Prometheus to scrape metrics off of `node` and `model_server`.
4. `grafana`: Add a Prometheus datasource and create a dashboard that autorefreshes every 5s and contains the following information. You may define the visualization type that would best represent the data.

   - Current timestamp
   - Operation Metrics
     - HTTP Success Rate
     - Rate of requests
     - Total requests in progress
     - Server Latency
     - Total requests made

   - Resource Metrics
     - Total processors
     - Total RAM
     - System load
     - Memory Usage
     - Disk I/O
     - Available disk space

   - Model Performance Metrics
     - Inference latency
     - Prediction distribution 

And finally, write a `locustfile.py` file to simulate the following conditions with appropriate weights. Your simulated production traffic should simulate at least 50 users for >3 mins.

- Request that responds with response code = 200
- Request that responds with response code = 4xx
- Request that responds with response code = 5xx

## More exercise
Here are some additional exercises that you may explore and implement.

1. Using Grafana provisioning to allow dashboard source control and ease container setups.
2. Explore on triggering alerts on Slack/ Telegram/ Discord when a certain metric is being anomalous
   - Using Prometheus Alertmanager 
   - Using Grafana Alerting