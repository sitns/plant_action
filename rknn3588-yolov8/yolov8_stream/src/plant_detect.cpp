/**
 * plant_detect.cpp
 * 植物检测主程序 - 使用 plant_det.rknn 模型
 * 支持：摄像头、图像文件、视频文件
 */

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <rknn_api.h>

#include "process/plant_postprocess.h"

#define INPUT_SIZE 640

// 获取当前时间（毫秒）
static double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Letterbox 预处理
static cv::Mat letterbox(const cv::Mat& src, int target_size, int& pad_x, int& pad_y, float& scale) {
    int src_w = src.cols;
    int src_h = src.rows;
    
    scale = std::min((float)target_size / src_w, (float)target_size / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    
    pad_x = (target_size - new_w) / 2;
    pad_y = (target_size - new_h) / 2;
    
    cv::Mat padded = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    cv::Rect roi(pad_x, pad_y, new_w, new_h);
    cv::resize(src, padded(roi), cv::Size(new_w, new_h));
    
    return padded;
}

// 运行推理
static int runInference(rknn_context ctx, const cv::Mat& img, 
                         const std::vector<std::string>& class_names,
                         rknn_tensor_type output_type,
                         int32_t output_zp,
                         float output_scale,
                         cv::Mat& result_img) {
    int pad_x, pad_y;
    float scale;
    
    // 先做 letterbox，再在 640x640 上做颜色转换，减少大图 cvtColor 开销。
    cv::Mat input_bgr = letterbox(img, INPUT_SIZE, pad_x, pad_y, scale);
    cv::Mat input;
    cv::cvtColor(input_bgr, input, cv::COLOR_BGR2RGB);
    
    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input.cols * input.rows * input.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input.data;
    
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        printf("设置输入失败: %d\n", ret);
        return -1;
    }
    
    // 运行推理
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("推理失败: %d\n", ret);
        return -1;
    }
    
    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = (output_type == RKNN_TENSOR_FLOAT16 || output_type == RKNN_TENSOR_INT8) ? 0 : 1;
    outputs[0].index = 0;
    
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("获取输出失败: %d\n", ret);
        return -1;
    }
    
    std::vector<PlantDetection> detections;
    if (output_type == RKNN_TENSOR_FLOAT16) {
        const uint16_t* output_data = static_cast<const uint16_t*>(outputs[0].buf);
        plantPostProcessFp16(output_data, class_names, detections);
    } else if (output_type == RKNN_TENSOR_INT8) {
        const int8_t* output_data = static_cast<const int8_t*>(outputs[0].buf);
        plantPostProcessInt8(output_data, output_zp, output_scale, class_names, detections);
    } else {
        float* output_data = static_cast<float*>(outputs[0].buf);
        plantPostProcessFast(output_data, class_names, detections);
    }
    
    // 缩放因子
    float scale_x = 1.0f / scale;
    float scale_y = 1.0f / scale;
    
    // 绘制结果
    result_img = img.clone();
    drawPlantDetections(result_img, detections, scale_x, scale_y, pad_x, pad_y, false);
    
    // 释放输出
    rknn_outputs_release(ctx, 1, outputs);
    
    return (int)detections.size();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("用法: %s <rknn_model> [image_path|camera_id] [--no-display]\n", argv[0]);
        printf("示例:\n");
        printf("  %s weights/plant_det.rknn 0          # 使用摄像头\n", argv[0]);
        printf("  %s weights/plant_det.rknn test.jpg    # 处理图像\n", argv[0]);
        printf("  %s weights/plant_det.rknn test.jpg --no-display  # 不显示窗口\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* input_source = (argc > 2) ? argv[2] : "0";
    
    // 检查是否有 --no-display 参数
    bool no_display = false;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--no-display") == 0) {
            no_display = true;
            break;
        }
    }
    
    // 加载类别
    std::vector<std::string> class_names;
    if (loadPlantClasses("src/coco_80_labels_list.txt", class_names) < 0) {
        // 尝试从其他路径加载
        if (loadPlantClasses("../../plant_det.txt", class_names) < 0) {
            printf("警告: 无法加载类别文件，将使用数字ID\n");
        }
    }
    
    // 读取模型
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("无法打开模型文件: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    uint8_t* model_data = (uint8_t*)malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);
    
    // 初始化 RKNN
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("RKNN 初始化失败: %d\n", ret);
        return -1;
    }
    printf("RKNN 模型加载成功\n");

    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret < 0) {
        printf("查询输出属性失败: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    printf("输出属性: type=%s, qnt=%s, zp=%d, scale=%f\n",
           get_type_string(output_attr.type),
           get_qnt_type_string(output_attr.qnt_type),
           output_attr.zp,
           output_attr.scale);
    
    // 判断输入源是摄像头还是文件
    int camera_id = atoi(input_source);
    bool is_camera = (strlen(input_source) == 1 && input_source[0] >= '0' && input_source[0] <= '9');
    
    cv::VideoCapture cap;
    if (is_camera) {
        // 使用 V4L2 后端打开摄像头
        std::string device_path = "/dev/video" + std::to_string(camera_id);
        cap.open(device_path, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        printf("使用摄像头 %s\n", device_path.c_str());
    } else {
        // 检查是否是图像文件
        std::string ext = input_source;
        size_t dot_pos = ext.find_last_of('.');
        if (dot_pos != std::string::npos) {
            ext = ext.substr(dot_pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        }
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            // 图像文件
            cv::Mat img = cv::imread(input_source);
            if (img.empty()) {
                printf("无法读取图像: %s\n", input_source);
                rknn_destroy(ctx);
                return -1;
            }
            
            printf("处理图像: %s (%dx%d)\n", input_source, img.cols, img.rows);
            
            double start_time = getCurrentTime();
            cv::Mat result;
            int num_det = runInference(ctx, img, class_names, output_attr.type, output_attr.zp, output_attr.scale, result);
            double elapsed = getCurrentTime() - start_time;
            
            printf("检测到 %d 个目标, 耗时: %.1f ms\n", num_det, elapsed);
            
            // 保存结果
            cv::imwrite("result.jpg", result);
            printf("结果已保存到 result.jpg\n");
            
            // 显示结果（如果没有 --no-display 参数）
            if (!no_display) {
                cv::imshow("Plant Detection", result);
                cv::waitKey(0);
            }
            
            rknn_destroy(ctx);
            return 0;
        } else {
            // 视频文件
            cap.open(input_source);
            printf("使用视频文件: %s\n", input_source);
        }
    }
    
    if (!cap.isOpened()) {
        printf("无法打开输入源\n");
        rknn_destroy(ctx);
        return -1;
    }
    
    printf("按 'q' 退出\n");
    
    int frame_count = 0;
    double total_time = 0;
    double fps = 0;
    
    printf("按 'q' 退出\n");
    
    // 创建显示窗口
    cv::namedWindow("Plant Detection", cv::WINDOW_AUTOSIZE);
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        double start_time = getCurrentTime();
        
        cv::Mat result;
        int num_det = runInference(ctx, frame, class_names, output_attr.type, output_attr.zp, output_attr.scale, result);
        
        double elapsed = getCurrentTime() - start_time;
        total_time += elapsed;
        frame_count++;
        fps = 1000.0 / elapsed;
        
        // 显示 FPS 和帧数
        char info[128];
        snprintf(info, sizeof(info), "FPS: %.1f | Frames: %d | Det: %d", fps, frame_count, num_det);
        cv::putText(result, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Plant Detection", result);
        
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;  // q 或 ESC 退出
    }
    
    if (frame_count > 0) {
        printf("\n平均 FPS: %.1f, 总帧数: %d\n", 1000.0 * frame_count / total_time, frame_count);
    }
    
    cap.release();
    cv::destroyAllWindows();
    rknn_destroy(ctx);
    
    return 0;
}
