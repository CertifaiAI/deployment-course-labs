# How Scheduler Works
There are 2 collections in mongodb that are related to a `drift_detector_server` container. The `image_features_sequence` collection serves as a counter of the number of documents in the `image_features` collection and record it in the `id` field.

![mongodb](https://user-images.githubusercontent.com/53246500/125891891-825f069c-1108-4f34-ad3e-756758de3fd0.png)

The scheduler pulls new data from mongodb and sends request(s) to the drift_detector_server if there is new data coming in. Following steps are how the scheduler works in each loop:

1. The scheduler gets the `id` from the `image_features_sequence` collection and assigns it as the number of documents in the `image_features` collection.

2. From the number of documents, it calculates the maximum number of batches to be the right bound of the `image_index` range in the `image_features` collection to be queried from (also the index of the lastest batch) by dividing the number of documents by the predefined batch size.

3. Then it resets the `image_index` range in the `image_features` collection to be queried from.
    - If there is no new data, the `image_index` range will be 0.

4. Next, it will check if there is any data in the `image_index` range which is updated in Step 3.
    - If there are data (new data) in the `image_index` range, it will send a request to drift_detector API `/driftdetect` endpoint to perform drift detection along with the image features queried from mongodb
        - After sending the request, the last and final step will be to update the `last_document_index` to be the left bound of the `image_index` range, this way in the new loop, only new data coming in will be posted to the drift_detector API endpoint.
    - If there is no data in the `image_index` range, it won't do anything and it will enter a new loop.




