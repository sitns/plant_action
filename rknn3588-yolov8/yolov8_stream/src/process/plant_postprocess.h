// plant_det 后处理 - 支持单输出 (1, 1123, 8400) 和 9 输出 DFL 格式

#ifndef PLANT_POSTPROCESS_H
#define PLANT_POSTPROCESS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#define PLANT_OBJ_THRESH 0.25f
#define PLANT_NMS_THRESH 0.45f
#define PLANT_IMG_SIZE 640
#define PLANT_NUM_CLASSES 1119
#define PLANT_NUM_BOXES 8400
#define PLANT_REG_MAX 16       // DFL reg_max (64 channels / 4 coords)
#define PLANT_NUM_HEADS 3      // 3 detection heads (stride 8/16/32)

struct PlantDetection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect box;
};

// 加载类别名称
int loadPlantClasses(const std::string& filename, std::vector<std::string>& classes);

// 原始后处理 (未优化)
int plantPostProcess(
    float* output,                              // 模型输出 [1123, 8400]
    const std::vector<std::string>& class_names,// 类别名称
    std::vector<PlantDetection>& detections,    // 检测结果
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 优化后处理 - float32 版本
// 使用缓存友好的循环顺序，便于编译器自动向量化
int plantPostProcessFast(
    float* output,                              // 模型输出 [1123, 8400], float32
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 优化后处理 - 原始 FP16 版本
// 直接读取 RKNN 输出的半精度数据，避免 runtime 扩成 float32
int plantPostProcessFp16(
    const uint16_t* output,                     // 模型输出 [1123, 8400], float16 raw bits
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 优化后处理 - INT8 版本 (跳过反量化)
// 直接在 INT8 空间做 argmax，避免 940万次 INT8→FP32 转换
// 需要配合 rknn_output.want_float = 0 使用
int plantPostProcessInt8(
    const int8_t* output,                       // 模型输出 [1123, 8400], int8
    int32_t zp,                                 // 量化零点
    float scale,                                // 量化缩放因子
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 9 输出 DFL 格式后处理 (多头检测模型)
// 输出布局: [reg8, cls8, extra8, reg16, cls16, extra16, reg32, cls32, extra32]
// reg: [64, H, W], cls: [num_classes, H, W], extra: [1, H, W] (ignored)
int plantPostProcess9Outputs(
    const float* outputs[9],                    // 9 个输出 tensor (float32)
    int num_outputs,                            // 输出数量 (9)
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 9 输出 DFL 格式后处理 - INT8 优化版
// 在 INT8 空间做 cls argmax，避免 ~940 万次 INT8→float32 反量化
// 4x 更少内存带宽 (INT8 vs float32)，缓存友好的 class-major 扫描
// 需要配合 rknn_output.want_float = 0 使用
int plantPostProcess9OutputsInt8(
    const int8_t* outputs[],                    // 9 个输出 tensor (INT8 raw)
    const int32_t zps[],                        // 每个输出的零点
    const float scales[],                       // 每个输出的缩放因子
    int num_outputs,                            // 输出数量 (9)
    const std::vector<std::string>& class_names,
    std::vector<PlantDetection>& detections,
    float obj_thresh = PLANT_OBJ_THRESH,
    float nms_thresh = PLANT_NMS_THRESH
);

// 在图像上绘制检测结果
// verbose: 是否打印每个检测到控制台 (实时路径应设为 false)
void drawPlantDetections(
    cv::Mat& img,
    const std::vector<PlantDetection>& detections,
    float scale_x = 1.0f,
    float scale_y = 1.0f,
    int pad_x = 0,
    int pad_y = 0,
    bool verbose = false,
    bool draw_label = true
);

#endif // PLANT_POSTPROCESS_H
