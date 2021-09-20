import sys
import pathlib
import io
import os
import cv2
import time
import torch
import imutils
import onnxruntime
import numpy as np
from PIL import Image
from rembg.bg import remove
import matplotlib.pyplot as plt
from imutils import perspective

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
from post_processing import get_post_processing


def preprocess_input(img_ori, short_size):
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_resize = resize_image(img, short_size)

    img = img_resize.astype(np.float32) / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img.astype(np.float32), 0), img_resize


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def cropping(path, height=400.0):
    r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
    with open(path, 'rb') as inputs:
        img = r(inputs)
        start = time.time()
        rs = remove(img)
        print('Cropping time:', time.time() - start)
    output = Image.open(io.BytesIO(rs)).convert("RGBA")
    img = np.array(output)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img.copy()

    ratio = img.shape[0] / height

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(gray, height=int(height))
    _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(threshold, 15)

    cnts, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    outline = None

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(polygon) == 4:
            outline = polygon.reshape(4, 2)

    if outline is None:
        result = orig
    else:
        result = perspective.four_point_transform(orig, outline * ratio)

    return result


class ONNXDetector:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        assert os.path.exists(model_path)
        ort_session = onnxruntime.InferenceSession(model_path)
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")

        self.model = ort_session
        config = {'type': 'SegDetectorRepresenter',
                  'args': {'thresh': 0.3, 'box_thresh': 0.7, 'max_candidates': 1000,
                           'unclip_ratio': 2}}
        self.post_process = get_post_processing(config)
        self.post_process.box_thresh = post_p_thre

    def infer(self, img_ori, short_size: int = 512):
        h, w = img_ori.shape[:2]
        batch = {'shape': [(h, w)]}

        input_predict, img_resize = preprocess_input(img_ori, short_size)

        ort_inputs = {self.model.get_inputs()[0].name: input_predict}
        ort_outs = self.model.run(None, ort_inputs)[0]
        tensor_out = torch.from_numpy(ort_outs)
        boxes_list, score_list = self.post_process(batch, tensor_out)
        boxes_list, score_list = boxes_list[0], score_list[0]
        if len(boxes_list) > 0:
            idx = boxes_list.reshape(boxes_list.shape[0], -1).sum(axis=1) > 0
            boxes_list, score_list = boxes_list[idx], score_list[idx]
        else:
            boxes_list, score_list = [], []

        return boxes_list, score_list

    @staticmethod
    def show(input_img, boxes_list):
        from utils.util import draw_bbox
        img = draw_bbox(input_img[:, :, ::-1], boxes_list)
        plt.figure(figsize=(8, 6), dpi=150)
        plt.imshow(img)


if __name__ == '__main__':
    onnx_path = 'detector.onnx'
    img_path = 'imgs/8Tq0Q.jpg'

    onnx_detector = ONNXDetector(onnx_path, post_p_thre=0.7, gpu_id=0)
    IMG = cropping(img_path)

    start = time.time()
    boxes_LST, score_LST = onnx_detector.infer(IMG)
    print('Infer time:', time.time() - start)

    # Show the result
    onnx_detector.show(IMG, boxes_LST)
