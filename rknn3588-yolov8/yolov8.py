import cv2
import numpy as np
from rknnlite.api import RKNNLite

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640


def load_classes(file_path):
    """从文件加载类别名称"""
    with open(file_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return tuple(classes)


CLASSES = load_classes('./plant_det.txt')
print(f"已加载 {len(CLASSES)} 个植物类别")


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e_y / np.sum(e_y, axis=2, keepdims=True)

    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    """
    处理 YOLOv8 模型输出
    输入格式: (1, 1123, 8400) 
    1123 = 4 (bbox: x, y, w, h) + 1119 (类别概率)
    8400 = 检测框数量
    """
    if len(input_data) != 1:
        print(f"警告: 期望1个输出，实际得到{len(input_data)}个")
        return None, None, None
    
    output = input_data[0]  # shape: (1, 1123, 8400)
    
    # 转置为 (1, 8400, 1123)
    output = output.transpose(0, 2, 1)
    
    # 分离边界框和类别概率
    bbox = output[0, :, :4]  # (8400, 4) - x, y, w, h 格式
    class_probs = output[0, :, 4:]  # (8400, 1119)
    
    # 计算每个框的最大类别分数
    class_max_score = np.max(class_probs, axis=-1)
    classes = np.argmax(class_probs, axis=-1)
    
    # 应用置信度阈值
    _class_pos = np.where(class_max_score >= OBJ_THRESH)
    
    if len(_class_pos[0]) == 0:
        return None, None, None
    
    scores = class_max_score[_class_pos]
    bbox = bbox[_class_pos]
    classes = classes[_class_pos]
    
    # 将中心点格式 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
    # 坐标已经是像素值，不需要再乘以 IMG_SIZE
    cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        cl = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(cl[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)


def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        top = (top - padding[0]) / ratio[0]
        left = (left - padding[1]) / ratio[1]
        right = (right - padding[0]) / ratio[0]
        bottom = (bottom - padding[1]) / ratio[1]

        top = int(top)
        left = int(left)

        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                   (top, left - 6),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 0, 255), 2)
        print(f"检测到: {CLASSES[cl]} 置信度: {score:.2f} 位置: ({top}, {left}, {int(right)}, {int(bottom)})")


def init_model(model_path, target='rk3588'):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("加载 RKNN 模型失败")
        return None

    if target == 'rk3588':
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn.init_runtime()

    if ret != 0:
        print("初始化运行环境失败")
        return None

    return rknn


def run_inference(rknn, img_path):
    img_src = cv2.imread(img_path)
    if img_src is None:
        print(f"无法读取图像: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_letterbox, ratio, padding = letterbox(img_rgb)
    img_input = np.expand_dims(img_letterbox, 0)

    outputs = rknn.inference(inputs=[img_input], data_format=['nhwc'])

    # 调试：打印输出形状
    print(f"模型输出数量: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"输出 {i} 形状: {out.shape}")

    boxes, classes, scores = yolov8_post_process(outputs)

    if boxes is not None and len(boxes) > 0:
        draw(img_src, boxes, scores, classes, ratio, padding)

    return img_src


if __name__ == '__main__':
    model_path = "./rknnModel/plant_det.rknn"
    img_path = "./planttest.jpg"
    target = "rk3588"

    print("=" * 50)
    print("Plant Detection - YOLOv8 RKNN Inference")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"设备: {target}")
    print(f"图像: {img_path}")
    print("=" * 50)

    rknn = init_model(model_path, target)
    if rknn is None:
        print("模型初始化失败")
        exit(-1)

    print("开始推理...")
    result_img = run_inference(rknn, img_path)

    if result_img is not None:
        import os
        if not os.path.exists('./result'):
            os.mkdir('./result')
        result_path = './result/planttest_result.jpg'
        cv2.imwrite(result_path, result_img)
        print(f"结果已保存到: {result_path}")
    else:
        print("推理失败")

    rknn.release()
    print("完成")
