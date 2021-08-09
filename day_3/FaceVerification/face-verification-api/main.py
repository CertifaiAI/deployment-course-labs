import uvicorn
from fastapi import FastAPI

from models import FaceVerifyRequest
from services import detect_face, get_face_embedding, calculate_distance

app = FastAPI()


@app.post("/verify")
def verify_face(request: FaceVerifyRequest):
    face1_image = detect_face(request.image1_string)
    face2_image = detect_face(request.image2_string)

    face1_embedding = get_face_embedding(face1_image)
    face2_embedding = get_face_embedding(face2_image)

    return {
        'distance': float(calculate_distance(face1_embedding, face2_embedding))
    }


if __name__ == '__main__':
    uvicorn.run(app)
