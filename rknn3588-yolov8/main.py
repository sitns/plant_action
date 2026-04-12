import cv2
import time
from rknnpool import rknnPoolExecutor
from func import myFunc

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit(-1)

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 读取第一帧确认摄像头工作正常
ret, frame = cap.read()
if not ret:
    print("无法读取摄像头画面")
    cap.release()
    exit(-1)

print(f"摄像头已连接，分辨率: {frame.shape[1]}x{frame.shape[0]}")

modelPath = "./rknnModel/plant_det.rknn"
# 线程数，每个 NPU 核心分配 2 个实例，共 6 个线程
TPEs = 6

# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# 初始化异步所需要的帧
for i in range(TPEs + 1):
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)

print(f"开始推理，使用 {TPEs} 个线程，按 'q' 退出...")

frames = 0
loopTime = time.time()
initTime = time.time()

while True:
    frames += 1
    
    # 从摄像头读取新帧
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break
    
    pool.put(frame)
    result, flag = pool.get()
    
    if not flag or result is None:
        continue
    
    # 计算并显示FPS
    if frames % 30 == 0:
        fps = 30 / (time.time() - loopTime)
        print(f"30帧平均帧率: {fps:.2f} FPS")
        loopTime = time.time()
    
    # 在图像上显示FPS
    current_fps = frames / (time.time() - initTime)
    cv2.putText(result, f'FPS: {current_fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, f'Frames: {frames}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Plant Detection', result)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n总平均帧率: {frames / (time.time() - initTime):.2f} FPS")
print(f"总帧数: {frames}")

# 释放资源
cap.release()
cv2.destroyAllWindows()
pool.release()
print("完成")
