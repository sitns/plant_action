from pathlib import Path

import cv2
import numpy as np

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640
BASE_DIR = Path(__file__).resolve().parent
CLASS_FILE = BASE_DIR / "plant_det.txt"


def load_classes(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        classes = [line.strip() for line in file.readlines() if line.strip()]
    return tuple(classes)


CLASSES = load_classes(CLASS_FILE)


def clamp(value, low, high):
    return max(low, min(high, value))


def scale_box_to_original(box, ratio, padding, image_shape):
    x1, y1, x2, y2 = box
    pad_x, pad_y = padding
    height, width = image_shape[:2]

    x1 = int(round((x1 - pad_x) / ratio[0]))
    y1 = int(round((y1 - pad_y) / ratio[1]))
    x2 = int(round((x2 - pad_x) / ratio[0]))
    y2 = int(round((y2 - pad_y) / ratio[1]))

    x1 = clamp(x1, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    x2 = clamp(x2, 0, width - 1)
    y2 = clamp(y2, 0, height - 1)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def build_top_candidates(class_probs, top_k=3):
    best_scores = np.max(class_probs, axis=0)
    top_indices = np.argsort(best_scores)[::-1][:top_k]
    candidates = []
    for index in top_indices:
        candidates.append(
            {
                "label": CLASSES[int(index)],
                "score": float(best_scores[index]),
                "class_id": int(index),
            }
        )
    return candidates


def build_detections(boxes, scores, classes, ratio, padding, image_shape):
    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = scale_box_to_original(box, ratio, padding, image_shape)
        detections.append(
            {
                "label": CLASSES[int(class_id)],
                "score": float(score),
                "class_id": int(class_id),
                "box": (x1, y1, x2, y2),
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


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
    return np.array(keep)


def yolov8_post_process(input_data):
    if len(input_data) != 1:
        print(f"警告: 期望1个输出，实际得到{len(input_data)}个")
        return None, None, None, []

    output = input_data[0].transpose(0, 2, 1)
    bbox = output[0, :, :4]
    class_probs = output[0, :, 4:]
    top_candidates = build_top_candidates(class_probs)

    class_max_score = np.max(class_probs, axis=-1)
    classes = np.argmax(class_probs, axis=-1)
    class_positions = np.where(class_max_score >= OBJ_THRESH)

    if len(class_positions[0]) == 0:
        return None, None, None, top_candidates

    scores = class_max_score[class_positions]
    bbox = bbox[class_positions]
    classes = classes[class_positions]

    cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    kept_boxes = []
    kept_classes = []
    kept_scores = []
    for class_id in set(classes):
        indices = np.where(classes == class_id)
        class_boxes = boxes[indices]
        class_scores = scores[indices]
        class_ids = classes[indices]
        keep = nms_boxes(class_boxes, class_scores)
        if len(keep) > 0:
            kept_boxes.append(class_boxes[keep])
            kept_classes.append(class_ids[keep])
            kept_scores.append(class_scores[keep])

    if not kept_boxes:
        return None, None, None, top_candidates

    return (
        np.concatenate(kept_boxes),
        np.concatenate(kept_classes),
        np.concatenate(kept_scores),
        top_candidates,
    )


def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_top = max(0, y - text_h - baseline - 10)
    box_bottom = max(text_h + baseline + 8, y)
    cv2.rectangle(
        image,
        (x, box_top),
        (x + text_w + 14, box_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 7, box_bottom - baseline - 4),
        font,
        scale,
        (245, 248, 240),
        thickness,
        cv2.LINE_AA,
    )


def draw(image, boxes, scores, classes, ratio, padding):
    detections = build_detections(boxes, scores, classes, ratio, padding, image.shape)
    colors = [
        (57, 130, 247),
        (46, 168, 120),
        (64, 89, 255),
        (33, 64, 25),
    ]

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        score = detection["score"]
        label = detection["label"]
        color = colors[index % len(colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        draw_label(image, f"{label} {score:.2f}", x1, y1, color)
    return detections


def letterbox(image, new_shape=(640, 640), color=(0, 0, 0)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio_value = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = ratio_value, ratio_value
    new_unpad = int(round(shape[1] * ratio_value)), int(round(shape[0] * ratio_value))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image, ratio, (left, top)


def detect_frame(rknn_lite, image):
    annotated = image.copy()
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image, ratio, padding = letterbox(input_image)
    input_image = np.expand_dims(input_image, 0)

    outputs = rknn_lite.inference(inputs=[input_image], data_format=["nhwc"])
    boxes, classes, scores, top_candidates = yolov8_post_process(outputs)

    detections = []
    if boxes is not None:
        detections = draw(annotated, boxes, scores, classes, ratio, padding)

    return {
        "image": annotated,
        "detections": detections,
        "top_candidates": top_candidates,
    }


def myFunc(rknn_lite, image):
    result = detect_frame(rknn_lite, image)
    return result["image"]
