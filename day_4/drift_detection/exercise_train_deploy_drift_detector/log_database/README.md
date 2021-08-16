# Log Database

This database is to log the input data that have been passed to the model. One of the reasons of having this database is so that we will be able to perform drift detection on the input data over a certain period of time. 

For the input data that we are logging, we expect to log extracted image features so you have to think about the representation data to be saved in the database.

You may use any database that you deem suitable to be the log database.