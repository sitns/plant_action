from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed


def initRKNN(rknnModel="./rknnModel/yolov5s.rknn", core_id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    
    # 根据 core_id 分配不同的 NPU 核心
    if core_id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    else:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(f"{rknnModel} NPU_CORE_{core_id} \t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="./rknnModel/yolov5s.rknn", TPEs=6):
    """初始化多个 RKNN 实例，循环绑定到 3 个 NPU 核心"""
    rknn_list = []
    for i in range(TPEs):
        core_id = i % 3  # 0, 1, 2 循环
        rknn_list.append(initRKNN(rknnModel, core_id))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, self.TPEs)
        self.pool = ThreadPoolExecutor(max_workers=self.TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
