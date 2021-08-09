# Face Verification API
Before running this example, please complete the steps below:

1. If you dont have onnxruntime and fastapi installed, please run `pip install -r requirements.txt` to install the required module.

2. Download the model weights below, create a folder with the name "weights" and put the downloaded model weights into the "weights" folder.

- [Face Detector](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/face-detector.onnx)
- [Face Embedding](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/mobile_arcface.onnx)

To start the application you can run `uvicorn main:app --reload`

The command `uvicorn main:app --reload` refers to:

- main: the file main.py (the Python "module").
- app: the object created inside of main.py with the line app = FastAPI().
- --reload: make the server restart after code changes. Only use for development.

Original pretrained models are taken from these repo:
- RetinaFace: https://github.com/biubug6/Pytorch_Retinaface
- ArcFace: https://github.com/TreB1eN/InsightFace_Pytorch