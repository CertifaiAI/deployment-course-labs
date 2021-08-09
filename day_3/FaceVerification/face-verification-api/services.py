import numpy as np
import onnxruntime as rt

import config as config
from utils import prior_box, decode, py_cpu_nms, base64_to_image

sess_face_detector = rt.InferenceSession(config.FACE_DETECTOR['model_path'])
face_detector_input_name = sess_face_detector.get_inputs()[0].name
face_detector_output_name = [o.name for o in sess_face_detector.get_outputs()]

sess_face_embedding = rt.InferenceSession(config.FACE_EMBEDDING['model_path'])
face_embedding_input_name = sess_face_embedding.get_inputs()[0].name
face_embedding_output_name = sess_face_embedding.get_outputs()[0].name


def detect_face(image_string):
    TARGET_SIZE = 640
    FACE_IMAGE_SIZE = 112
    image = base64_to_image(image_string)

    img_resized = image.resize((TARGET_SIZE, TARGET_SIZE))
    img_resized = np.array(img_resized)

    im_height, im_width, _ = img_resized.shape

    scale = np.array([im_width, im_height, im_width, im_height])

    img = np.float32(img_resized)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    loc, conf, landms = sess_face_detector.run(face_detector_output_name, {face_detector_input_name: img})
    priorbox = prior_box((im_height, im_width), config.FACE_DETECTOR['steps'], config.FACE_DETECTOR['clip'],
                         config.FACE_DETECTOR['min_sizes'])
    boxes = decode(loc.squeeze(0), priorbox, config.FACE_DETECTOR['variance'])
    boxes = boxes * scale
    scores = conf.squeeze(0)[:, 1]

    # ignore low scores
    inds = np.where(scores > config.FACE_DETECTOR['confidence_threshold'])[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:config.FACE_DETECTOR['top_k']]
    boxes = boxes[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, config.FACE_DETECTOR['nms_threshold'])
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:config.FACE_DETECTOR['keep_top_k'], :]
    dets = dets[dets[:, 4] > 0.6, :]

    dets[:, 0] = np.round(dets[:, 0] * (image.width / TARGET_SIZE))
    dets[:, 1] = np.round(dets[:, 1] * (image.height / TARGET_SIZE))
    dets[:, 2] = np.round(dets[:, 2] * (image.width / TARGET_SIZE))
    dets[:, 3] = np.round(dets[:, 3] * (image.height / TARGET_SIZE))

    box_order = np.argsort((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[::-1]
    dets = dets[box_order]

    # select largest
    largest_face = dets[0, :4].astype(int)
    face_image = image.crop([largest_face[0], largest_face[1], largest_face[2], largest_face[3]])
    face_image = face_image.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
    return face_image


def get_face_embedding(face_image):
    img = np.float32(face_image)
    img = img/255
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    face_embedding = sess_face_embedding.run([face_embedding_output_name], {face_embedding_input_name: img})[0]

    return face_embedding

def calculate_distance(face_embedding_1, face_embedding_2):
    return np.linalg.norm(face_embedding_1-face_embedding_2)