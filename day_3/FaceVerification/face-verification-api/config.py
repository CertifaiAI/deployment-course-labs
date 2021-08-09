
FACE_DETECTOR = {
    'model_path': './weights/face-detector.onnx',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'clip': False,
    'variance': [0.1, 0.2],
    'confidence_threshold': 0.02,
    'top_k': 5000,
    'keep_top_k': 750,
    'nms_threshold': 0.4
}

FACE_EMBEDDING = {
    'model_path': './weights/mobile_arcface.onnx'
}