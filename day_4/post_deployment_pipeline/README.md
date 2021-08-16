# Post Deployment Pipeline
1. [Download Models](#download-models)
2. [Viewing the Dashboard](#viewing-the-dashboard)
3. [Simulating Production Traffic](#simulating-production-traffic)
4. [Perform AB Testing](#perform-ab-testing)
5. [Bringing Up the Containers](#bringing-up-the-containers)
6. [Port Mapping](#port-mapping)
7. [Important Notes](#important-notes)
8. [Known Issues](#known-issues)

## Download Models

The required files to run this demo is stored in cloud. You can manually download them from the links below and place them in the `app/artifacts/` folder in each of these folders:

- [Drift detector](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/drift_detector.pkl) for `drift_detector_server` and `drift_detector_server_new`
- [Fruit classifier model](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt) for `model_server`
- [New transfer learning model](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/transfer_learning_model.pt) & [fruit classifier model](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt) for `model_server_new`

Alternatively, you can just run the download_models.py script. The script will create the respective `artifacts/` folder for you and save the downloaded model into the folder.
```
python download_models.py
```

## Bringing Up the Containers
- Run `docker-compose up` in this directory; 
- Run it in `docker-compose up -d` if you want to run it in detached mode with no verbose output.
- Run `docker-compose up --build` if you want to rebuild the Docker images for the containers
- `CTRL+C` to stop the containers.
- To remove the containers after stopping the containers, run `docker-compose down` 
- To remove the mongodb volume as you remove the containers, run `docker-compose down -v`

## Architecture

<center>
    
![PDP Architecture](https://user-images.githubusercontent.com/76937732/129270279-46936b25-5487-4d1c-a5bd-dfc111d55047.png)
    
</center>

## Viewing the Dashboard

1. After creating the containers with `docker-compose up`, go to the following link.

    ```
    http://localhost:3000
    ```
2. Use 'admin' as both your username and password. Click 'Skip'.
3. At the navigation bar on the left, hover over 'Dashboards' (four squares) and select 'Manage'. 
4. There are two dashboard choices here. Click on the monitoring dashboard that you would like to view.

## Simulating Production Traffic

1. Change your working directory to *./locust*
2. Create a virtual environment.

    ```
    python3 -m venv venv
    ```
3. Activate the virtual environment.

    ```
    source venv/bin/activate
    ```
4. Check if your environment was activated properly. The following command should result in the directory of your new *venv* folder.

    ```
    which pip
    ```
5. Install the dependencies for testing.

    ```
    pip install -r requirements.txt
    ```
6. Once that's done, run the following command to begin loading traffic. By default, the nginx server at *http://localhost:80* is loaded by simulating 100 users for 5 minutes. You may configure these settings in *./locust.conf*.

    ```
    locust --config locust.conf
    ```
7. Alternatively, if you preferred having a UI, run the following command instead and head to *http://localhost:8089*. Select the number of users to simulate, spawn rate and the target address with `http://[HOST]:[PORT_NUMBER]`

    ```
    locust -f locustfile.py
    ```
8. For users with **Windows machines that has Anaconda/ Miniconda installed**, you may run the following code line by line to create an environment called `locustwin`, activate it and install the dependencies. The commands to begin loading traffic is the same as Linux.

    ```
    conda create -n locustwin python==3.8.10
    conda activate locustwin
    pip install -r requirements_win.txt
    ```

## Perform AB Testing

In order to perform AB Testing, you can just change the machine learning model in `model_server_new/`, as it serves as the new container that is going to serve that model.

You can also configure routing rules that suit to your needs in `nginx/default.conf`. The setting now is that each request will be passed to a different server in a sequential manner.


## Port Mapping
- 8000 => `model_server`
- 8001 => `model_server_new`
- 80 =>   `nginx`
- 8009 => `drift_detector_server`
- 8010 => `drift_detector_server_new`
- 9090 => `prometheus`
- 9091 => `prometheus_new`
- 3000 => `grafana`

## Important Notes
- `mongo_data` in the mongodb folder is a persistent bind-mount for the mongodb database. It is mostly for backup purpose, however  since what we are demonstrating here on Grafana is a real time visualisation, it doesn't make sense to display the past p-value from drift detection. Hence you can delete the folder if you want to restart the containers and perform `docker volume rm VOLUME_NAME`

- How the scheduler works in `drift_detector_server` and `drift_detector_server_new` can be found in the README in the `drift_detector_server/scheduler` and `drift_detector_server_new/scheduler` folders respectively.
## Known Issues
[TRANSFER TO KB SOON]
1.  Unable to make queries to Prometheus server due to time difference between container and OS. </br>
    **Signs:**</br>
    > a. Dashboard in Grafana loads but there is no graph output </br>
    > b. Error message in *http://localhost:9090* as follows:-</br>

    ![Time difference](https://user-images.githubusercontent.com/76937732/125857960-272653b6-b3a0-49d4-ae90-e20531010968.png)
    
    **OS:**</br>
    >Windows 10 Professional WSL2 </br>

    **Solution:**</br>
    >Run the following in bash:-

    ```
    sudo apt-get update
    sudo apt-get install ntpdate
    sudo ntpdate pool.ntp.org
    ```

2. Bash script converted to CRLF line endings in Windows
    If you're using Docker for Windows, you might run into this error:
    > drift_detector_server | start.sh: line 2: $'\r': command not found

    ![CRLF Line Endings](https://user-images.githubusercontent.com/53246500/125881034-1cea00bd-4112-43b1-a38a-39ef73abb23e.png)

    This is due to Windows text editor changing the bash script to a DOS-style CRLF line endings which is absent in the Unix-style line endings (LF). 

    **Solution:**</br>
    Inside WSL, run:
    ```
    sudo apt-get install dos2unix
    dos2unix drift_detector_server
    ```
