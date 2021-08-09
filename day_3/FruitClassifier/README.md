# Running Guide
## Fruit Classifier
Step 1. Create conda environment.\
Open terminal from directory `deployment-course-labs`, run the command below. Skip this step if the environment
has been created.
```
conda env create -f environment.yml
```

Step 2. Activate conda environment.\
Run the command below.
```
conda activate deploy
```

Step 3. Change directory to `day_3/FruitClassifier/ModelConversion` using command below.
```
cd day_3/FruitClassifier/ModelConversion
```

Step 4. Run script to convert model into torchscript.
```
python trace_model.py
```

Step 5. Open `day3/FruitClassifier/App` in Android Studio to run the application.

Step 6. Run the phone application by following the [guide](../README.md#application-running)