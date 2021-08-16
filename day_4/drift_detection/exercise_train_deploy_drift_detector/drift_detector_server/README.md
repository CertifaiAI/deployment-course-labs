# Drift Detector Server

You may also use Fast API to deploy the drift detector, remember that you have to load the serialised trained drift detector file to perform drift detection.

## Hint: 

- In order for the drift detection result to be meaningful, it can only be performed on a batch of data. In other words, you have to accumulate at least a number of input data to be able to perform drift detection.

- Try to think about what are the appropriate query parameters to be sent to the drift detector server, eg:
    - timeframe
    - p-value threshold
    - batch size

