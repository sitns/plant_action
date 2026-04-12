/**
 * plant_detect_thread.cpp
 * 植物检测主程序 - 多线程版本
 * 每个线程绑定独立的 NPU 核心，提高推理速度
 */

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <atomic>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <rknn_api.h>

#include "process/plant_postprocess.h"

#define INPUT_SIZE 640
#define NUM_NPU_CORES 3
#define NUM_THREADS 6  // 每个核心2个线程

// 获取当前时间（毫秒）
static double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static std::string getDisplayClassName(const PlantDetection& det, const std::vector<std::string>& class_names) {
    if (!det.class_name.empty()) {
        return det.class_name;
    }
    if (det.class_id >= 0 && det.class_id < (int)class_names.size() && !class_names[det.class_id].empty()) {
        return class_names[det.class_id];
    }
    return "id#" + std::to_string(det.class_id);
}

static void printDetectionSummary(const std::vector<PlantDetection>& detections,
                                  double fps,
                                  const std::vector<std::string>& class_names) {
    struct ClassSummary {
        int count = 0;
        float best_confidence = 0.0f;
    };

    std::unordered_map<std::string, ClassSummary> class_stats;
    class_stats.reserve(detections.size());
    for (const auto& det : detections) {
        ClassSummary& summary = class_stats[getDisplayClassName(det, class_names)];
        ++summary.count;
        if (det.confidence > summary.best_confidence) {
            summary.best_confidence = det.confidence;
        }
    }

    std::vector<std::pair<std::string, ClassSummary>> sorted_counts(class_stats.begin(), class_stats.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const std::pair<std::string, ClassSummary>& a, const std::pair<std::string, ClassSummary>& b) {
                  if (a.second.count != b.second.count) {
                      return a.second.count > b.second.count;
                  }
                  if (a.second.best_confidence != b.second.best_confidence) {
                      return a.second.best_confidence > b.second.best_confidence;
                  }
                  return a.first < b.first;
              });

    printf("\rFPS: %.1f | Detections: %zu | Classes: ", fps, detections.size());
    if (sorted_counts.empty()) {
        printf("none                      ");
    } else {
        const size_t max_classes_to_print = 5;
        for (size_t i = 0; i < sorted_counts.size() && i < max_classes_to_print; ++i) {
            if (i > 0) {
                printf(", ");
            }
            printf("%s(%d,%.2f)",
                   sorted_counts[i].first.c_str(),
                   sorted_counts[i].second.count,
                   sorted_counts[i].second.best_confidence);
        }
        if (sorted_counts.size() > max_classes_to_print) {
            printf(", ...");
        }
        printf("            ");
    }
    fflush(stdout);
}

static void printTopDetections(const std::vector<PlantDetection>& detections,
                               const std::vector<std::string>& class_names,
                               size_t max_items = 10) {
    if (detections.empty()) {
        printf("未检测到植物类别\n");
        return;
    }

    std::vector<const PlantDetection*> sorted;
    sorted.reserve(detections.size());
    for (const auto& det : detections) {
        sorted.push_back(&det);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const PlantDetection* a, const PlantDetection* b) {
                  return a->confidence > b->confidence;
              });

    printf("检测结果列表:\n");
    for (size_t i = 0; i < sorted.size() && i < max_items; ++i) {
        const PlantDetection& det = *sorted[i];
        const std::string class_name = getDisplayClassName(det, class_names);
        printf("  %zu. %s | conf=%.2f | box=[%d,%d,%d,%d]\n",
               i + 1,
               class_name.c_str(),
               det.confidence,
               det.box.x,
               det.box.y,
               det.box.width,
               det.box.height);
    }
    if (sorted.size() > max_items) {
        printf("  ... 其余 %zu 个结果省略\n", sorted.size() - max_items);
    }
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

// 性能分析开关: 1=打印 preprocess/inference/postprocess 耗时
#define PROFILE_TIMING 1

// RKNN 推理上下文
struct RKNNContext {
    rknn_context ctx;
    int core_id;
    bool initialized;
    int n_outputs;              // 输出 tensor 数量 (1 或 9)
    // 每个输出的量化参数
    int32_t output_zps[9];
    float output_scales[9];
    uint32_t output_sizes[9];   // 每个输出的字节大小 (INT8)
    rknn_tensor_type output_type;
    rknn_tensor_qnt_type output_qnt_type;
    // 预分配输出缓冲区 (避免每帧 malloc/free)
    uint8_t* output_bufs[9];
};

// 推理任务
struct InferenceTask {
    int id;
    cv::Mat frame;
    std::vector<PlantDetection> detections;
    bool completed;
};

// 全局变量
static RKNNContext g_contexts[NUM_THREADS];
static std::vector<std::string> g_class_names;
static uint8_t* g_model_data = nullptr;
static int g_model_size = 0;

// 线程池
class ThreadPool {
private:
    std::queue<InferenceTask*> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<std::thread> workers;
    std::atomic<bool> stop;
    
    // 结果队列
    std::map<int, InferenceTask*> result_map;
    std::mutex result_mutex;
    std::condition_variable result_cv;
    int next_result_id;
    
    void workerFunc(int thread_id) {
        RKNNContext& ctx = g_contexts[thread_id];
        
        while (!stop) {
            InferenceTask* task = nullptr;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return !task_queue.empty() || stop; });
                
                if (stop) break;
                if (task_queue.empty()) continue;
                
                task = task_queue.front();
                task_queue.pop();
            }
            
            if (task) {
                // 运行推理
                runInference(ctx, task->frame, task->detections);
                task->completed = true;
                
                // 将结果放入结果map
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    result_map[task->id] = task;
                }
                result_cv.notify_one();
            }
        }
    }
    
    void runInference(RKNNContext& ctx, const cv::Mat& img, 
                      std::vector<PlantDetection>& detections) {
        int pad_x, pad_y;
        float scale;
        
#if PROFILE_TIMING
        double t0 = getCurrentTime();
#endif
        // 先做 letterbox，再在 640x640 上做颜色转换，减少大图 cvtColor 开销。
        cv::Mat input_bgr = letterbox(img, INPUT_SIZE, pad_x, pad_y, scale);
        cv::Mat input;
        cv::cvtColor(input_bgr, input, cv::COLOR_BGR2RGB);
        
#if PROFILE_TIMING
        double t1 = getCurrentTime();
#endif
        // 设置输入
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = input.cols * input.rows * input.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input.data;
        
        int ret = rknn_inputs_set(ctx.ctx, 1, inputs);
        if (ret < 0) return;
        
        // 运行推理
        ret = rknn_run(ctx.ctx, NULL);
        if (ret < 0) return;
        
        if (ctx.n_outputs >= 6 && ctx.output_type == RKNN_TENSOR_INT8) {
            // === INT8 多输出优化路径 ===
            // want_float=0: 跳过 ~1000 万次 INT8→float32 反量化
            // is_prealloc=1: 避免每帧 malloc/free ~10MB
            const int n = ctx.n_outputs;
            rknn_output outputs[9];
            memset(outputs, 0, sizeof(outputs));
            for (int i = 0; i < n; ++i) {
                outputs[i].index = i;
                outputs[i].want_float = 0;      // 保持 INT8 原始数据!
                outputs[i].is_prealloc = 1;      // 使用预分配缓冲区
                outputs[i].buf = ctx.output_bufs[i];
                outputs[i].size = ctx.output_sizes[i];
            }
            
            ret = rknn_outputs_get(ctx.ctx, n, outputs, NULL);
            if (ret < 0) return;
            
#if PROFILE_TIMING
            double t2 = getCurrentTime();
#endif
            const int8_t* out_ptrs[9];
            for (int i = 0; i < n; ++i) {
                out_ptrs[i] = static_cast<const int8_t*>(outputs[i].buf);
            }
            
            plantPostProcess9OutputsInt8(out_ptrs, ctx.output_zps, ctx.output_scales,
                                         n, g_class_names, detections);
            
            rknn_outputs_release(ctx.ctx, n, outputs);

#if PROFILE_TIMING
            double t3 = getCurrentTime();
            static thread_local int prof_count = 0;
            if (++prof_count % 100 == 1) {
                printf("\n[Profile] preproc=%.1fms infer+get=%.1fms postproc=%.1fms total=%.1fms\n",
                       t1-t0, t2-t1, t3-t2, t3-t0);
            }
#endif
        } else if (ctx.n_outputs >= 6) {
            // 多输出 float 回退路径
            const int n = ctx.n_outputs;
            rknn_output outputs[9];
            memset(outputs, 0, sizeof(outputs));
            for (int i = 0; i < n; ++i) {
                outputs[i].index = i;
                outputs[i].want_float = 1;
            }
            
            ret = rknn_outputs_get(ctx.ctx, n, outputs, NULL);
            if (ret < 0) return;
            
#if PROFILE_TIMING
            double t2 = getCurrentTime();
#endif
            const float* out_ptrs[9];
            for (int i = 0; i < n; ++i) {
                out_ptrs[i] = static_cast<const float*>(outputs[i].buf);
            }
            
            plantPostProcess9Outputs(out_ptrs, n, g_class_names, detections);
            
            rknn_outputs_release(ctx.ctx, n, outputs);

#if PROFILE_TIMING
            double t3 = getCurrentTime();
            static thread_local int prof_count2 = 0;
            if (++prof_count2 % 100 == 1) {
                printf("\n[Profile/float] preproc=%.1fms infer+get=%.1fms postproc=%.1fms total=%.1fms\n",
                       t1-t0, t2-t1, t3-t2, t3-t0);
            }
#endif
        } else {
            // 单输出模型 (1, 1123, 8400)
            rknn_output outputs[1];
            memset(outputs, 0, sizeof(outputs));
            outputs[0].want_float = (ctx.output_type == RKNN_TENSOR_FLOAT16 || ctx.output_type == RKNN_TENSOR_INT8) ? 0 : 1;
            outputs[0].index = 0;
            
            ret = rknn_outputs_get(ctx.ctx, 1, outputs, NULL);
            if (ret < 0) return;
            
            if (ctx.output_type == RKNN_TENSOR_FLOAT16) {
                const uint16_t* output_data = static_cast<const uint16_t*>(outputs[0].buf);
                plantPostProcessFp16(output_data, g_class_names, detections);
            } else if (ctx.output_type == RKNN_TENSOR_INT8) {
                const int8_t* output_data = static_cast<const int8_t*>(outputs[0].buf);
                plantPostProcessInt8(output_data, ctx.output_zps[0], ctx.output_scales[0], g_class_names, detections);
            } else {
                float* output_data = static_cast<float*>(outputs[0].buf);
                plantPostProcessFast(output_data, g_class_names, detections);
            }
            
            rknn_outputs_release(ctx.ctx, 1, outputs);
        }
        
        // 缩放回原图 (与 func.py draw 函数逻辑一致: (coord - pad) / scale)
        for (auto& det : detections) {
            int x = (int)((det.box.x - pad_x) / scale);
            int y = (int)((det.box.y - pad_y) / scale);
            int w = (int)(det.box.width / scale);
            int h = (int)(det.box.height / scale);
            det.box = cv::Rect(x, y, w, h);
        }
    }
    
public:
    ThreadPool() : stop(false), next_result_id(0) {}
    
    ~ThreadPool() {
        stopAll();
    }
    
    void start(int num_threads) {
        for (int i = 0; i < num_threads; i++) {
            workers.emplace_back(&ThreadPool::workerFunc, this, i);
        }
        printf("线程池启动: %d 个线程\n", num_threads);
    }
    
    void submitTask(InferenceTask* task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
        }
        queue_cv.notify_one();
    }
    
    bool getResult(int id, InferenceTask*& task, bool blocking = true) {
        std::unique_lock<std::mutex> lock(result_mutex);
        
        if (blocking) {
            result_cv.wait(lock, [this, id] { 
                return result_map.find(id) != result_map.end(); 
            });
        } else {
            if (result_map.find(id) == result_map.end()) {
                return false;
            }
        }
        
        auto it = result_map.find(id);
        if (it != result_map.end()) {
            task = it->second;
            result_map.erase(it);
            return true;
        }
        return false;
    }
    
    void stopAll() {
        stop = true;
        queue_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
    }
};

// 初始化 RKNN 上下文
static int initRKNNContext(RKNNContext& ctx, const char* model_data, int model_size, int core_id) {
    memset(&ctx, 0, sizeof(ctx));

    int ret = rknn_init(&ctx.ctx, (void*)model_data, model_size, 0, NULL);
    if (ret < 0) {
        printf("RKNN 初始化失败: %d\n", ret);
        return -1;
    }
    
    // 设置 NPU 核心
    rknn_core_mask core_mask;
    switch (core_id % NUM_NPU_CORES) {
        case 0: core_mask = RKNN_NPU_CORE_0; break;
        case 1: core_mask = RKNN_NPU_CORE_1; break;
        case 2: core_mask = RKNN_NPU_CORE_2; break;
        default: core_mask = RKNN_NPU_CORE_0; break;
    }
    
    ret = rknn_set_core_mask(ctx.ctx, core_mask);
    if (ret < 0) {
        printf("设置 NPU 核心 %d 失败: %d\n", core_id, ret);
    } else {
        printf("RKNN 上下文 %d 绑定到 NPU 核心 %d\n", core_id, core_id % NUM_NPU_CORES);
    }

    // 查询输入输出数量
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx.ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("查询输入输出数量失败: %d\n", ret);
        return -1;
    }
    ctx.n_outputs = std::min(static_cast<int>(io_num.n_output), 9);
    printf("模型输出数量: %d\n", ctx.n_outputs);

    // 查询所有输出属性 (量化参数 + 缓冲区大小)
    for (int i = 0; i < ctx.n_outputs; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx.ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret < 0) {
            printf("查询输出[%d]属性失败: %d\n", i, ret);
            return -1;
        }
        ctx.output_zps[i] = attr.zp;
        ctx.output_scales[i] = attr.scale;
        ctx.output_sizes[i] = attr.n_elems;  // INT8: 1 byte per element

        if (i == 0) {
            ctx.output_type = attr.type;
            ctx.output_qnt_type = attr.qnt_type;
            printf("输出[0]属性: type=%s, qnt=%s, zp=%d, scale=%f\n",
                   get_type_string(attr.type),
                   get_qnt_type_string(attr.qnt_type),
                   attr.zp, attr.scale);
        }
    }

    // 预分配输出缓冲区 (INT8 模型的多输出路径)
    if (ctx.n_outputs >= 6 && ctx.output_type == RKNN_TENSOR_INT8) {
        for (int i = 0; i < ctx.n_outputs; ++i) {
            ctx.output_bufs[i] = (uint8_t*)malloc(ctx.output_sizes[i]);
            if (!ctx.output_bufs[i]) {
                printf("分配输出缓冲区[%d] (%u bytes) 失败\n", i, ctx.output_sizes[i]);
                return -1;
            }
        }
    }
    
    ctx.core_id = core_id % NUM_NPU_CORES;
    ctx.initialized = true;
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("用法: %s <rknn_model> [image_path|camera_id]\n", argv[0]);
        printf("示例:\n");
        printf("  %s weights/plant_det.rknn 0          # 使用摄像头\n", argv[0]);
        printf("  %s weights/plant_det.rknn test.jpg    # 处理图像\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* input_source = (argc > 2) ? argv[2] : "0";
    
    // 加载类别
    if (loadPlantClasses("plant_det.txt", g_class_names) < 0) {
        if (loadPlantClasses("../../plant_det.txt", g_class_names) < 0) {
            printf("警告: 无法加载类别文件\n");
        }
    }
    
    // 读取模型文件
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("无法打开模型文件: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    g_model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    g_model_data = (uint8_t*)malloc(g_model_size);
    fread(g_model_data, 1, g_model_size, fp);
    fclose(fp);
    
    // 初始化多个 RKNN 上下文
    for (int i = 0; i < NUM_THREADS; i++) {
        if (initRKNNContext(g_contexts[i], (const char*)g_model_data, g_model_size, i % NUM_NPU_CORES) < 0) {
            printf("初始化 RKNN 上下文 %d 失败\n", i);
            return -1;
        }
    }
    
    // 启动线程池
    ThreadPool pool;
    pool.start(NUM_THREADS);
    
    // 判断输入源
    int camera_id = atoi(input_source);
    bool is_camera = (strlen(input_source) == 1 && input_source[0] >= '0' && input_source[0] <= '9');
    
    cv::VideoCapture cap;
    if (is_camera) {
        // 使用 GStreamer 管道打开摄像头
        // 使用 MJPG 格式打开摄像头 (720p)
        std::string pipeline = "v4l2src device=/dev/video" + std::to_string(camera_id) + 
                               " ! image/jpeg,width=1280,height=720,framerate=30/1"
                               " ! jpegdec ! videoconvert ! appsink";
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            // 备用方式：直接使用设备号，设置 720p MJPG
            cap.open(camera_id, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 30);
        }
        printf("使用摄像头 %d\n", camera_id);
    } else {
        std::string ext = input_source;
        size_t dot_pos = ext.find_last_of('.');
        if (dot_pos != std::string::npos) {
            ext = ext.substr(dot_pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        }
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            // 图像文件 - 单帧处理
            cv::Mat img = cv::imread(input_source);
            if (img.empty()) {
                printf("无法读取图像: %s\n", input_source);
                return -1;
            }
            
            printf("处理图像: %s (%dx%d)\n", input_source, img.cols, img.rows);
            
            double start_time = getCurrentTime();
            
            InferenceTask task;
            task.id = 0;
            task.frame = img;
            task.completed = false;
            
            pool.submitTask(&task);
            InferenceTask* result = nullptr;
            pool.getResult(0, result);
            
            double elapsed = getCurrentTime() - start_time;
            
            printf("检测到 %zu 个目标, 耗时: %.1f ms\n", result->detections.size(), elapsed);
            printDetectionSummary(result->detections, elapsed > 0.0 ? 1000.0 / elapsed : 0.0, g_class_names);
            printf("\n");
            printTopDetections(result->detections, g_class_names);
            
            // 绘制结果
            cv::Mat result_img = img.clone();
            drawPlantDetections(result_img, result->detections, 1.0f, 1.0f, 0, 0, false, true);
            
            cv::imwrite("result.jpg", result_img);
            printf("结果已保存到 result.jpg\n");
            
            return 0;
        } else {
            cap.open(input_source);
            printf("使用视频文件: %s\n", input_source);
        }
    }
    
    if (!cap.isOpened()) {
        printf("无法打开输入源\n");
        return -1;
    }
    
    // 创建 GUI 窗口
    const char* win_name = "Plant Detection (Multi-Thread)";
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    
    printf("开始实时推理，按 'q' 键退出...\n");
    
    int frame_count = 0;
    int task_id = 0;
    double start_time = getCurrentTime();
    double fps = 0.0;
    double last_gui_update = 0.0;
    double last_summary_update = 0.0;
    const double gui_update_interval_ms = 66.0;
    const double summary_update_interval_ms = 200.0;
    
    // 流水线：先填满线程池
    int pipeline_depth = NUM_THREADS;
    int submitted = 0;
    int retrieved = 0;
    
    // 预提交 pipeline_depth 帧
    for (int i = 0; i < pipeline_depth; i++) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        InferenceTask* task = new InferenceTask();
        task->id = task_id++;
        task->frame = frame;
        task->completed = false;
        pool.submitTask(task);
        submitted++;
    }
    
    while (true) {
        // 提交新帧
        cv::Mat frame;
        cap >> frame;
        if (!frame.empty()) {
            InferenceTask* task = new InferenceTask();
            task->id = task_id++;
            task->frame = frame;
            task->completed = false;
            pool.submitTask(task);
            submitted++;
        }
        
        // 阻塞获取最早提交的结果
        InferenceTask* result = nullptr;
        if (pool.getResult(retrieved, result, true)) {
            retrieved++;
            frame_count++;
            
            // 计算 FPS
            double now = getCurrentTime();
            fps = frame_count * 1000.0 / (now - start_time);

            if (now - last_summary_update >= summary_update_interval_ms) {
                printDetectionSummary(result->detections, fps, g_class_names);
                last_summary_update = now;
            }

            if (now - last_gui_update >= gui_update_interval_ms) {
                cv::Mat& display = result->frame;
                drawPlantDetections(display, result->detections, 1.0f, 1.0f, 0, 0, false, false);

                char fps_text[64];
                snprintf(fps_text, sizeof(fps_text), "FPS: %.1f | Detections: %zu", fps, result->detections.size());
                cv::putText(display, fps_text, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                cv::imshow(win_name, display);
                last_gui_update = now;
            }
            
            delete result;
        }
        
        // 检测按键，'q' 退出
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q' || key == 27) {
            printf("用户退出\n");
            break;
        }
        
        // 视频文件播放完毕
        if (frame.empty() && retrieved >= submitted) break;
    }
    
    double elapsed = getCurrentTime() - start_time;
    printf("\n平均 FPS: %.1f, 总帧数: %d\n", 1000.0 * frame_count / elapsed, frame_count);
    
    pool.stopAll();
    cap.release();
    cv::destroyAllWindows();
    
    for (int i = 0; i < NUM_THREADS; i++) {
        if (g_contexts[i].initialized) {
            // 释放预分配的输出缓冲区
            for (int j = 0; j < g_contexts[i].n_outputs; ++j) {
                if (g_contexts[i].output_bufs[j]) {
                    free(g_contexts[i].output_bufs[j]);
                    g_contexts[i].output_bufs[j] = nullptr;
                }
            }
            rknn_destroy(g_contexts[i].ctx);
        }
    }
    free(g_model_data);
    
    return 0;
}
