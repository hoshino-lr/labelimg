import cv2 as cv
import numpy as np


class net_detect(object):
    _cls2name = {
        0: "car",
        1: "watcher",
        2: "base"
    }

    def __init__(self, path):
        self.__init_ok = False
        # net1参数
        self.net1_confThreshold = 0.3
        self.net1_nmsThreshold = 0.4
        self.net1_inpHeight = 640
        self.net1_inpWidth = 640

        # 不检测base

        self.net1_grid = []
        self.net1_num_anchors = [3, 3, 3]
        self.net1_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198],
                             [373, 326]]
        self.net1_strides = [8, 16, 32]
        for i in self.net1_strides:
            self.net1_grid.append([self.net1_num_anchors[0], int(self.net1_inpHeight / i), int(self.net1_inpWidth / i)])

        try:
            self.net = cv.dnn.readNetFromONNX(path)  # 加载训练好的识别模型
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            self.__init_ok = True

        except Exception as e:
            print("error：" + str(e))

    def update(self, img: np.ndarray):
        if not self.__init_ok or img is None:
            return []
        draw = img.copy()
        image_height, image_width, _ = img.shape
        self.x_factor = image_width / self.net1_inpWidth
        self.y_factor = image_height / self.net1_inpHeight
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (self.net1_inpWidth, self.net1_inpHeight), swapRB=True, crop=False)
        self.net.setInput(blob)  # 设置模型输入
        out = self.net.forward()  # 推理出结果
        return self._net1_process(out)

    def _net1_process(self, output):
        # 第一个网络的处理

        # :param res 输出为一个(N,6)的array 或为一个空array

        classIds = []
        confidences = []
        bboxes = []
        output = output.reshape(-1, 8)
        choose = output[:, 4] > self.net1_confThreshold
        output = output[choose]
        choose = np.where(choose == True)[0]
        for i in range(0, len(choose)):
            xc = output[i, :]
            max_id = np.argmax(xc[5:])  # 选择置信度最高的 class
            obj_conf = float(xc[4] * xc[5 + max_id])  # 置信度

            x, y, w, h = xc[0].item(), xc[1].item(), xc[2].item(), xc[3].item()
            left = int((x - 0.5 * w) * self.x_factor)
            top = int((y - 0.5 * h) * self.y_factor)
            width = int(w * self.x_factor)
            height = int(h * self.y_factor)

            bboxes.append([left, top, width, height])
            classIds.append(max_id)
            confidences.append(obj_conf)

        # NMS筛选
        indices = cv.dnn.NMSBoxes(bboxes, confidences, self.net1_confThreshold, self.net1_nmsThreshold)
        res = []

        if len(indices):
            for i in indices:
                x = bboxes[i][0]
                y = bboxes[i][1]
                w = bboxes[i][2]
                h = bboxes[i][3]
                bbox = {
                    "difficult": False,
                    "label": self._cls2name[classIds[i]],
                    "points": [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
                }
                res.append(bbox)

        return res
