// plant_det 后处理实现 - 支持单输出 (1, 1123, 8400) 和 9 输出 DFL 格式

#include "plant_postprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace {

inline float fp16ToFloat(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;

    if (exp == 0) {
        if (mant == 0) {
            const uint32_t bits = sign;
            float out;
            memcpy(&out, &bits, sizeof(out));
            return out;
        }
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            --exp;
        }
        ++exp;
        mant &= ~0x0400U;
    } else if (exp == 31) {
        const uint32_t bits = sign | 0x7f800000U | (mant << 13);
        float out;
        memcpy(&out, &bits, sizeof(out));
        return out;
    }

    exp = exp + (127 - 15);
    const uint32_t bits = sign | (exp << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

inline float dequantizeInt8(int8_t value, int32_t zp, float scale) {
    return (static_cast<int32_t>(value) - zp) * scale;
}

inline cv::Rect toClampedRect(float cx, float cy, float w, float h) {
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(PLANT_IMG_SIZE)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(PLANT_IMG_SIZE)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(PLANT_IMG_SIZE)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(PLANT_IMG_SIZE)));

    return cv::Rect(
        static_cast<int>(x1),
        static_cast<int>(y1),
        static_cast<int>(std::max(0.0f, x2 - x1)),
        static_cast<int>(std::max(0.0f, y2 - y1))
    );
}

inline void fillDetection(
    int box_index,
    int class_id,
    float confidence,
    const std::vector<std::string>& class_names,
    const float* x_ptr,
    const float* y_ptr,
    const float* w_ptr,
    const float* h_ptr,
    std::vector<PlantDetection>& detections
) {
    PlantDetection det;
    det.class_id = class_id;
    det.class_name = class_id < static_cast<int>(class_names.size()) ? class_names[class_id] : "unknown";
    det.confidence = confidence;
    det.box = toClampedRect(x_ptr[box_index], y_ptr[box_index], w_ptr[box_index], h_ptr[box_index]);
    detections.push_back(std::move(det));
}

inline void fillDetectionInt8(
    int box_index,
    int class_id,
    float confidence,
    int32_t zp,
    float scale,
    const std::vector<std::string>& class_names,
    const int8_t* x_ptr,
    const int8_t* y_ptr,
    const int8_t* w_ptr,
    const int8_t* h_ptr,
    std::vector<PlantDetection>& detections
) {
    PlantDetection det;
    det.class_id = class_id;
    det.class_name = class_id < static_cast<int>(class_names.size()) ? class_names[class_id] : "unknown";
    det.confidence = confidence;
    det.box = toClampedRect(
        dequantizeInt8(x_ptr[box_index], zp, scale),
        dequantizeInt8(y_ptr[box_index], zp, scale),
        dequantizeInt8(w_ptr[box_index], zp, scale),
        dequantizeInt8(h_ptr[box_index], zp, scale)
    );
    detections.push_back(std::move(det));
}

float calcIoU(const cv::Rect& a, const cv::Rect& b) {
    const float inter_area = static_cast<float>((a & b).area());
    const float union_area = static_cast<float>(a.area() + b.area()) - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

void nms(std::vector<PlantDetection>& detections, float nms_thresh) {
    std::sort(detections.begin(), detections.end(),
              [](const PlantDetection& a, const PlantDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<uint8_t> suppressed(detections.size(), 0);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].class_id != detections[j].class_id) {
                continue;
            }
            if (calcIoU(detections[i].box, detections[j].box) > nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }

    size_t write_index = 0;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i]) {
            detections[write_index++] = std::move(detections[i]);
        }
    }
    detections.resize(write_index);
}

}  // namespace

int loadPlantClasses(const std::string& filename, std::vector<std::string>& classes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("无法打开类别文件: %s\n", filename.c_str());
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == ' ' || line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            classes.push_back(line);
        }
    }

    printf("加载了 %zu 个植物类别\n", classes.size());
    return 0;
}

int plantPostProcess(
    float* output,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    return plantPostProcessFast(output, class_names, detections, obj_thresh, nms_thresh);
}

int plantPostProcessFast(
    float* output,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    const int class_count = std::min(PLANT_NUM_CLASSES, static_cast<int>(class_names.size()));
    const float* x_ptr = output;
    const float* y_ptr = output + PLANT_NUM_BOXES;
    const float* w_ptr = output + 2 * PLANT_NUM_BOXES;
    const float* h_ptr = output + 3 * PLANT_NUM_BOXES;

    detections.clear();
    detections.reserve(256);

    std::vector<float> best_scores(PLANT_NUM_BOXES, -std::numeric_limits<float>::infinity());
    std::vector<int> best_classes(PLANT_NUM_BOXES, -1);

    // 输出布局是 [attr][box]，按类别顺序扫描能连续访问内存，避免每次跨 8400 步长。
    for (int c = 0; c < class_count; ++c) {
        const float* class_ptr = output + (4 + c) * PLANT_NUM_BOXES;
        for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
            const float score = class_ptr[i];
            if (score > best_scores[i]) {
                best_scores[i] = score;
                best_classes[i] = c;
            }
        }
    }

    for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
        if (best_scores[i] < obj_thresh || best_classes[i] < 0) {
            continue;
        }
        fillDetection(i, best_classes[i], best_scores[i], class_names, x_ptr, y_ptr, w_ptr, h_ptr, detections);
    }

    nms(detections, nms_thresh);
    return 0;
}

int plantPostProcessFp16(
    const uint16_t* output,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    const int class_count = std::min(PLANT_NUM_CLASSES, static_cast<int>(class_names.size()));
    const uint16_t* x_ptr = output;
    const uint16_t* y_ptr = output + PLANT_NUM_BOXES;
    const uint16_t* w_ptr = output + 2 * PLANT_NUM_BOXES;
    const uint16_t* h_ptr = output + 3 * PLANT_NUM_BOXES;

    detections.clear();
    detections.reserve(256);

    std::vector<float> best_scores(PLANT_NUM_BOXES, -std::numeric_limits<float>::infinity());
    std::vector<int> best_classes(PLANT_NUM_BOXES, -1);

    for (int c = 0; c < class_count; ++c) {
        const uint16_t* class_ptr = output + (4 + c) * PLANT_NUM_BOXES;
        for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
            const float score = fp16ToFloat(class_ptr[i]);
            if (score > best_scores[i]) {
                best_scores[i] = score;
                best_classes[i] = c;
            }
        }
    }

    for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
        if (best_scores[i] < obj_thresh || best_classes[i] < 0) {
            continue;
        }

        PlantDetection det;
        det.class_id = best_classes[i];
        det.class_name = det.class_id < static_cast<int>(class_names.size()) ? class_names[det.class_id] : "unknown";
        det.confidence = best_scores[i];
        det.box = toClampedRect(
            fp16ToFloat(x_ptr[i]),
            fp16ToFloat(y_ptr[i]),
            fp16ToFloat(w_ptr[i]),
            fp16ToFloat(h_ptr[i])
        );
        detections.push_back(std::move(det));
    }

    nms(detections, nms_thresh);
    return 0;
}

int plantPostProcessInt8(
    const int8_t* output,
    int32_t zp,
    float scale,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    const int class_count = std::min(PLANT_NUM_CLASSES, static_cast<int>(class_names.size()));
    const int8_t* x_ptr = output;
    const int8_t* y_ptr = output + PLANT_NUM_BOXES;
    const int8_t* w_ptr = output + 2 * PLANT_NUM_BOXES;
    const int8_t* h_ptr = output + 3 * PLANT_NUM_BOXES;
    const int8_t threshold_q = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(obj_thresh / scale + zp))));

    detections.clear();
    detections.reserve(256);

    std::vector<int8_t> best_scores(PLANT_NUM_BOXES, std::numeric_limits<int8_t>::min());
    std::vector<int> best_classes(PLANT_NUM_BOXES, -1);

    for (int c = 0; c < class_count; ++c) {
        const int8_t* class_ptr = output + (4 + c) * PLANT_NUM_BOXES;
        for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
            const int8_t score_q = class_ptr[i];
            if (score_q > best_scores[i]) {
                best_scores[i] = score_q;
                best_classes[i] = c;
            }
        }
    }

    for (int i = 0; i < PLANT_NUM_BOXES; ++i) {
        if (best_scores[i] < threshold_q || best_classes[i] < 0) {
            continue;
        }
        const float confidence = dequantizeInt8(best_scores[i], zp, scale);
        if (confidence < obj_thresh) {
            continue;
        }
        fillDetectionInt8(i, best_classes[i], confidence, zp, scale, class_names, x_ptr, y_ptr, w_ptr, h_ptr, detections);
    }

    nms(detections, nms_thresh);
    return 0;
}

// ===== 9 输出 DFL 格式后处理 =====

int plantPostProcess9Outputs(
    const float* outputs[9],
    int num_outputs,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    detections.clear();
    detections.reserve(256);

    const int num_classes = std::min(PLANT_NUM_CLASSES, static_cast<int>(class_names.size()));
    // 3 heads, each has 3 outputs: reg, cls, extra
    // outputs_per_head can be 3 (9 outputs) or 2 (6 outputs)
    const int outputs_per_head = num_outputs / PLANT_NUM_HEADS;
    const int grid_sizes[] = {80, 40, 20};
    const int strides[] = {8, 16, 32};

    for (int head = 0; head < PLANT_NUM_HEADS; ++head) {
        const float* reg = outputs[head * outputs_per_head + 0];  // [64, H, W]
        const float* cls = outputs[head * outputs_per_head + 1];  // [num_classes, H, W]

        const int gh = grid_sizes[head];
        const int gw = grid_sizes[head];
        const int area = gh * gw;
        const int stride = strides[head];

        // DFL decode: [64, H, W] -> softmax over reg_max -> weighted sum -> [4, H, W]
        // Then convert to xyxy pixel coordinates using grid + stride

        for (int i = 0; i < area; ++i) {
            // Quick cls argmax for this grid cell
            float best_score = -1.0f;
            int best_cls = -1;
            for (int c = 0; c < num_classes; ++c) {
                const float score = cls[c * area + i];
                if (score > best_score) {
                    best_score = score;
                    best_cls = c;
                }
            }

            if (best_score < obj_thresh || best_cls < 0) {
                continue;
            }

            // DFL decode for this grid cell only (avoid decoding all 8400 boxes)
            const int gy = i / gw;
            const int gx = i % gw;
            float dist[4];  // left, top, right, bottom distances

            for (int p = 0; p < 4; ++p) {
                // softmax over PLANT_REG_MAX bins
                float max_val = -1e30f;
                for (int k = 0; k < PLANT_REG_MAX; ++k) {
                    float v = reg[(p * PLANT_REG_MAX + k) * area + i];
                    if (v > max_val) max_val = v;
                }
                float sum_exp = 0.0f;
                float expected = 0.0f;
                for (int k = 0; k < PLANT_REG_MAX; ++k) {
                    float e = expf(reg[(p * PLANT_REG_MAX + k) * area + i] - max_val);
                    sum_exp += e;
                    expected += e * k;
                }
                dist[p] = expected / sum_exp;
            }

            float x1 = (gx + 0.5f - dist[0]) * stride;
            float y1 = (gy + 0.5f - dist[1]) * stride;
            float x2 = (gx + 0.5f + dist[2]) * stride;
            float y2 = (gy + 0.5f + dist[3]) * stride;

            // clamp
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(PLANT_IMG_SIZE)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(PLANT_IMG_SIZE)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(PLANT_IMG_SIZE)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(PLANT_IMG_SIZE)));

            PlantDetection det;
            det.class_id = best_cls;
            det.class_name = best_cls < static_cast<int>(class_names.size()) ? class_names[best_cls] : "unknown";
            det.confidence = best_score;
            det.box = cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(std::max(0.0f, x2 - x1)),
                static_cast<int>(std::max(0.0f, y2 - y1))
            );
            detections.push_back(std::move(det));
        }
    }

    nms(detections, nms_thresh);
    return 0;
}

// ===== 9 输出 DFL 格式后处理 - INT8 优化版 =====
// 核心优化:
// 1. 在 INT8 空间做 cls argmax，避免 ~940 万次反量化
// 2. 4x 更少内存带宽 (INT8 vs float32)
// 3. 只对通过阈值的 cell 做 DFL 解码 (懒解码)
// 4. Class-major 扫描顺序，缓存友好

int plantPostProcess9OutputsInt8(
    const int8_t* outputs[],
    const int32_t zps[],
    const float scales[],
    int num_outputs,
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh,
    float nms_thresh
) {
    detections.clear();
    detections.reserve(256);

    const int num_classes = std::min(PLANT_NUM_CLASSES, static_cast<int>(class_names.size()));
    const int outputs_per_head = num_outputs / PLANT_NUM_HEADS;
    const int grid_sizes[] = {80, 40, 20};
    const int strides[] = {8, 16, 32};

    for (int head = 0; head < PLANT_NUM_HEADS; ++head) {
        const int reg_idx = head * outputs_per_head;
        const int cls_idx = head * outputs_per_head + 1;

        const int8_t* reg = outputs[reg_idx];   // [64, H, W]
        const int8_t* cls = outputs[cls_idx];   // [num_classes, H, W]

        const int32_t cls_zp = zps[cls_idx];
        const float cls_scale = scales[cls_idx];
        const int32_t reg_zp = zps[reg_idx];
        const float reg_scale = scales[reg_idx];

        const int gh = grid_sizes[head];
        const int gw = grid_sizes[head];
        const int area = gh * gw;
        const int stride = strides[head];

        // 将 float 阈值转换为 INT8 空间
        int32_t thresh_i32 = static_cast<int32_t>(std::round(obj_thresh / cls_scale + cls_zp));
        int8_t int8_thresh = static_cast<int8_t>(std::max(-128, std::min(127, thresh_i32)));

        // === 快速死头检测 (仅对大 grid) ===
        // stride 8 (area=6400) 和 stride 16 (area=1600) 的量化后死头全是常数
        // 跳过可节省 ~7ms + ~2ms，stride 32 (area=400) 计算量小，总是处理
        if (area >= 1600) {
            const int total = num_classes * area;
            const int8_t v = cls[0];
            if (v == cls[area - 1] &&
                v == cls[total / 2] &&
                v == cls[total - 1] &&
                v == cls[area / 2] &&
                v == cls[num_classes / 2 * area] &&
                v < int8_thresh) {
                continue;  // 跳过死头
            }
        }

        // Class-major INT8 argmax (缓存友好: cls 布局是 [num_classes, H, W])
        // 每个 class 的数据在内存中是连续的 → 顺序读取，充分利用 cache line
        int8_t best_scores[6400];   // max area = 80*80
        int16_t best_classes[6400];
        memset(best_scores, 0x80, area);  // INT8_MIN = -128
        for (int i = 0; i < area; ++i) best_classes[i] = -1;

        for (int c = 0; c < num_classes; ++c) {
            const int8_t* cls_row = cls + c * area;  // 连续内存!
            for (int i = 0; i < area; ++i) {
                if (cls_row[i] > best_scores[i]) {
                    best_scores[i] = cls_row[i];
                    best_classes[i] = static_cast<int16_t>(c);
                }
            }
        }

        // 只处理通过阈值的 cell (通常只有几个)
        for (int i = 0; i < area; ++i) {
            if (best_scores[i] < int8_thresh || best_classes[i] < 0) continue;

            // 反量化置信度 (只对通过阈值的做)
            float confidence = (static_cast<int32_t>(best_scores[i]) - cls_zp) * cls_scale;
            if (confidence < obj_thresh) continue;

            // DFL 解码 - 只对通过阈值的 cell (懒解码)
            const int gy = i / gw;
            const int gx = i % gw;
            float dist[4];

            for (int p = 0; p < 4; ++p) {
                // 在 INT8 空间找 max (数值稳定性)
                int8_t max_q = -128;
                for (int k = 0; k < PLANT_REG_MAX; ++k) {
                    int8_t v = reg[(p * PLANT_REG_MAX + k) * area + i];
                    if (v > max_q) max_q = v;
                }
                // Softmax: 利用 INT8 差值 × scale 计算, 避免完整反量化
                float sum_exp = 0.0f, expected = 0.0f;
                for (int k = 0; k < PLANT_REG_MAX; ++k) {
                    float e = expf(static_cast<float>(
                        reg[(p * PLANT_REG_MAX + k) * area + i] - max_q) * reg_scale);
                    sum_exp += e;
                    expected += e * k;
                }
                dist[p] = expected / sum_exp;
            }

            float x1 = (gx + 0.5f - dist[0]) * stride;
            float y1 = (gy + 0.5f - dist[1]) * stride;
            float x2 = (gx + 0.5f + dist[2]) * stride;
            float y2 = (gy + 0.5f + dist[3]) * stride;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(PLANT_IMG_SIZE)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(PLANT_IMG_SIZE)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(PLANT_IMG_SIZE)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(PLANT_IMG_SIZE)));

            PlantDetection det;
            det.class_id = best_classes[i];
            det.class_name = det.class_id < static_cast<int>(class_names.size())
                             ? class_names[det.class_id] : "unknown";
            det.confidence = confidence;
            det.box = cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(std::max(0.0f, x2 - x1)),
                static_cast<int>(std::max(0.0f, y2 - y1))
            );
            detections.push_back(std::move(det));
        }
    }

    nms(detections, nms_thresh);
    return 0;
}

void drawPlantDetections(
    cv::Mat& img,
    const std::vector<PlantDetection>& detections,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y,
    bool verbose,
    bool draw_label
) {
    for (const auto& det : detections) {
        int x;
        int y;
        int w;
        int h;
        if (scale_x != 1.0f || scale_y != 1.0f || pad_x != 0 || pad_y != 0) {
            x = static_cast<int>((det.box.x - pad_x) / scale_x);
            y = static_cast<int>((det.box.y - pad_y) / scale_y);
            w = static_cast<int>(det.box.width / scale_x);
            h = static_cast<int>(det.box.height / scale_y);
        } else {
            x = det.box.x;
            y = det.box.y;
            w = det.box.width;
            h = det.box.height;
        }

        cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

        if (draw_label) {
            char label[256];
            snprintf(label, sizeof(label), "%s %.2f", det.class_name.c_str(), det.confidence);

            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(img,
                          cv::Rect(x, y - text_size.height - 4, text_size.width + 4, text_size.height + 4),
                          cv::Scalar(0, 255, 0), -1);
            cv::putText(img, label, cv::Point(x + 2, y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        if (verbose) {
            printf("检测到: %s 置信度: %.2f\n", det.class_name.c_str(), det.confidence);
        }
    }
}
