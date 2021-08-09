from fastai.vision.all import load_learner

def get_classifier():
    learner = load_learner('/opt/airflow/scripts/dog_classifier.pkl')
    return learner