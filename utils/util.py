# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:59
# @Author  : zhoujun
import os
import cv2
import json
import math
import pathlib
import time
import glob
import torch
import onnxruntime
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

from models import Model


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


def setup_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('DBNet.pytorch')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img, result, color=(255, 0, 0), thickness=2, is_sorted=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    if len(result) == 0:
        return img
    result = list(reversed(result))
    result = sort_bbox(result)
    for i, points in enumerate(result):
        points = points.astype(int)
        cv2.polylines(img, [points], True, color, thickness, lineType=cv2.LINE_AA)
        if is_sorted:
            cv2.putText(img, str(i), tuple(points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                        color, thickness, lineType=cv2.LINE_AA)
    return img


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def save_result(result_path, box_list, score_list, is_output_polygon):
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    """
    对只有一个字符的框进行扩充
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def distance(p1, p2, p):
    return abs(((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0]) /
               math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2))


def antipodal_pairs(convex_polygon):
    l = []
    n = len(convex_polygon)
    p1, p2 = convex_polygon[0], convex_polygon[1]

    t, d_max = None, 0
    for p in range(1, n):
        d = distance(p1, p2, convex_polygon[p])
        if d > d_max:
            t, d_max = p, d
    l.append(t)

    for p in range(1, n):
        p1, p2 = convex_polygon[p % n], convex_polygon[(p + 1) % n]
        _p, _pp = convex_polygon[t % n], convex_polygon[(t + 1) % n]
        while distance(p1, p2, _pp) > distance(p1, p2, _p):
            t = (t + 1) % n
            _p, _pp = convex_polygon[t % n], convex_polygon[(t + 1) % n]
        l.append(t)

    return l


# returns score, area, points from top-left, clockwise , favouring low area
def mep(convex_polygon, img_ori):
    def compute_parallelogram(convex_polygon, l, z1, z2):
        def parallel_vector(a, b, c):
            v0 = [c[0] - a[0], c[1] - a[1]]
            v1 = [b[0] - c[0], b[1] - c[1]]
            return [c[0] - v0[0] - v1[0], c[1] - v0[1] - v1[1]]

        # finds intersection between lines, given 2 points on each line.
        # (x1, y1), (x2, y2) on 1st line, (x3, y3), (x4, y4) on 2nd line.
        def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            return px, py

        # from each antipodal point, draw a parallel vector,
        # so ap1->ap2 is parallel to p1->p2
        #    aq1->aq2 is parallel to q1->q2
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1 + 1) % n]
        q1, q2 = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n]
        ap1, aq1 = convex_polygon[l[z1 % n]], convex_polygon[l[z2 % n]]
        ap2, aq2 = parallel_vector(p1, p2, ap1), parallel_vector(q1, q2, aq1)

        a = line_intersection(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1], q2[0], q2[1])
        b = line_intersection(p1[0], p1[1], p2[0], p2[1], aq1[0], aq1[1], aq2[0], aq2[1])
        d = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], q1[0], q1[1], q2[0], q2[1])
        c = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], aq1[0], aq1[1], aq2[0], aq2[1])

        s = distance(a, b, c) * math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        return s, a, b, c, d

    z1, z2 = 0, 0
    n = len(convex_polygon)

    # for each edge, find antipodal vertices for it (step 1 in paper).
    l = antipodal_pairs(convex_polygon)

    so, ao, bo, co, do, z1o, z2o = 100000000000, None, None, None, None, None, None

    # step 2 in paper.
    for z1 in range(0, n):
        if z1 >= z2:
            z2 = z1 + 1
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1 + 1) % n]
        a, b, c = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n], convex_polygon[l[z2 % n]]
        if distance(p1, p2, a) >= distance(p1, p2, b):
            continue

        while distance(p1, p2, c) > distance(p1, p2, b):
            z2 += 1
            a, b, c = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n], convex_polygon[
                l[z2 % n]]

        st, at, bt, ct, dt = compute_parallelogram(convex_polygon, l, z1, z2)

        # img_ori_ = img_ori.copy()
        # vis_bbox(img_ori_, [(int(at[0]), int(at[1])), (int(bt[0]), int(bt[1])),
        #                     (int(ct[0]), int(ct[1])), (int(dt[0]), int(dt[1]))])
        # cv2.destroyAllWindows()

        if st < so:
            so, ao, bo, co, do, z1o, z2o = st, at, bt, ct, dt, z1, z2

    return so, ao, bo, co, do, z1o, z2o


def detect_pre_process(img_ori, short_size):
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_resize = resize_image(img, short_size)

    img = img_resize.astype(np.float32) / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img.astype(np.float32), 0)


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


def load_detector(path, device, mode='torch'):
    net, post_process_config = None, None
    if mode == 'torch':
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        net = Model(config['arch']).to(device)
        net.load_state_dict(checkpoint['state_dict'])
        net.to(device)
        net.eval()
        post_process_config = config['post_processing']
    elif mode == 'onnx':
        post_process_config = {'type': 'SegDetectorRepresenter',
                               'args': {'thresh': 0.3, 'box_thresh': 0.7, 'max_candidates': 1000,
                                        'unclip_ratio': 2.5}}
        net = onnxruntime.InferenceSession(path)

    return net, post_process_config


def detect(net, item, short_size, device, post_processor, mode='onnx'):
    h, w = item.shape[:2]
    batch = {'shape': [(h, w)]}
    input_predict = detect_pre_process(item, short_size)

    tensor_out = None
    if mode == 'torch':
        input_predict = torch.Tensor(input_predict).to(device)
        if device == 'cuda':
            torch.cuda.synchronize(device)

        tensor_out = net(input_predict)

        if device == 'cuda':
            torch.cuda.synchronize(device)
    elif mode == 'onnx':
        ort_inputs = {net.get_inputs()[0].name: input_predict}
        ort_outs = net.run(None, ort_inputs)[0]
        tensor_out = torch.from_numpy(ort_outs)

    boxes_list, score_list = post_processor(batch, tensor_out)
    boxes_list, score_list = boxes_list[0], score_list[0]
    if len(boxes_list) > 0:
        idx = boxes_list.reshape(boxes_list.shape[0], -1).sum(axis=1) > 0
        boxes_list, score_list = boxes_list[idx], score_list[idx]
    else:
        boxes_list, score_list = [], []

    return boxes_list, score_list


def save_image(folder_path, file_name, image):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = os.path.join(folder_path, file_name)
    cv2.imwrite(save_path, image)


def on_one_line(box1, box2):
    off_h = min(box1[3][1] - box1[0][1], box2[3][1] - box2[0][1]) * 0.45  # 0.43 --> 0.47
    if np.abs(box1[0][1] - box2[0][1]) < off_h:
        return True
    return False


def sort_bbox(box_lst):
    result_dict = {0: [0]}
    line = 0
    for i, box in enumerate(box_lst[1:]):
        if on_one_line(box, box_lst[i]):
            result_dict[line].append(i + 1)
        else:
            line += 1
            result_dict[line] = [i + 1]

    result_lst = []
    for k, v in result_dict.items():
        if len(v) == 1:
            result_lst.append(box_lst[v[0]])
        else:
            line = sorted(box_lst[v[0]:v[-1] + 1], key=lambda x: x[0][0])
            result_lst.extend(line)

    return result_lst


def intersection_of_2lines(line1, line2):
    a1 = line1[0] - line1[2]
    b1 = line1[1] - line1[3]
    a2 = line2[0] - line2[2]
    b2 = line2[1] - line2[3]
    factor_matrix = [[b1, b2],
                     [-a1, -a2],
                     [b1 * line1[0] - a1 * line1[1], b2 * line2[0] - a2 * line2[1]]]

    D = np.linalg.det(factor_matrix[:2])
    Dx = np.linalg.det([factor_matrix[2], factor_matrix[0]])
    Dy = np.linalg.det([factor_matrix[1], factor_matrix[2]])
    x = Dx / D
    y = Dy / D

    return int(-y), int(-x)
