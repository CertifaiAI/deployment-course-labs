#
#################################################################################
#
#  Copyright (c) 2021 CertifAI Sdn. Bhd.
#
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################
#
import time
import math
import pickle
import torch
import requests
import json
import logging
import sys
import jsonpickle
from pymongo import MongoClient
from config import BATCH_SIZE, P_VALUE_THRESHOLD, DRIFT_DETECTOR_SERVICE

# every 2 seconds query from mongodb database
def mongo_auto_query():
    index_range = [0] * 2
    last_document_index = 0
    n_total_documents = 0

    # connect to mongodb
    # Uncomment line with host='localhost' when testing
    mongodbClient = MongoClient(host='mongodb',port=27017,username='admin', password='admin', authSource='drift_detection')
    # mongodbClient = MongoClient(host='localhost',port=27017,username='admin', password='admin', authSource='drift_detection')
    mongodb = mongodbClient["drift_detection"]

    while(True):
        # 1. get total number of documents (get id of image_features_sequence) 
        all_documents = mongodb.image_features_sequence.find({}, {'id':1, '_id':0})
        for x in all_documents:
            n_total_documents = x['id']
        logging.info("Total documents: " + str(n_total_documents))

        # 2. get number of batches available (each batch has to achieve BATCH_SIZE to be included)
        n_batch = math.floor(n_total_documents / BATCH_SIZE)

        # 3. reset index_range to query from [last_document_index, max index (n_batch)]
        index_range[0] = last_document_index
        index_range[1] = n_batch * BATCH_SIZE

        # 4. Query from start index to end index + send request to drift_detector API endpoint
        myquery = {'image_index': {'$gt':index_range[0], '$lt':index_range[-1] +1 }}
        
        # if there are image_features within the index_range, query image_features and send request to drift_detector API endpoint
        if mongodb.image_features.count_documents(myquery) != 0:
            logging.info("document exists")
        
            mydoc = mongodb.image_features.find(myquery)
            
            # unpickle into tensor
            tensor_list = [pickle.loads(x['image_features']) for x in mydoc]
        
            # concat tensors based on batch_size 
            # tensor_list[i:i+BATCH_SIZE] separates all the queried tensors into batches
            # torch.cat() concat each of the tensors in each batch
            list_of_batch_tensors = [torch.cat(tensor_list[i:i+BATCH_SIZE], dim=0) for i in range(0, len(tensor_list), BATCH_SIZE)]

            for batch_tensor in list_of_batch_tensors:
                # send request to drift_detector API endpoint
                logging.info(batch_tensor.shape)
                drift_detection_req = json.dumps({
                    "p_value_threshold": P_VALUE_THRESHOLD,
                    "batch_tensor": jsonpickle.encode(batch_tensor.detach().numpy())
                })
                response = requests.post("{}/driftdetect".format(DRIFT_DETECTOR_SERVICE), data=drift_detection_req)
                result = response.json()
                logging.info("Response headers: ", response.headers)
                logging.info("Result: ", result)

            # 5. assign last_document_index = right bound of index_range
            last_document_index = index_range[-1]

        else:
            logging.info("no new batch")
        
        # sleep 2 seconds before performing query again
        time.sleep(2)

if __name__ == '__main__': 
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    mongo_auto_query()