import os
from itertools import cycle

import cv2
# import h5py
import numpy as np
import torch
from numpy import linalg as LA

from IntegrateApp.models import UNet
# from 鼠标框选 import selectBoxByMouse
# from 多目标跟踪_re import mainTrack
from IntegrateApp import global_var

global NormPoint, sizeRate, winH, winW, imgH, imgW, normTargetFeatPoint, myThreshold
global imgMatrix
global stepSizeFeat
stepSizeFeat = {
    'stepH': 4,
    'stepW': 4
}
myThreshold = 0.75


def selectBoxByMouse(img):
    (min_x, min_y, width, height) = cv2.selectROI('MultiTracker', img)
    print('左上角坐标', (min_x, min_y))
    NormPoint = (min_x / img.shape[1], min_y / img.shape[0])
    print('左上角坐标比例，区间在(0,1)', NormPoint)
    print('高和宽', (height, width))
    rate = (width / img.shape[1], height / img.shape[0])
    print('高和宽归一化', rate)
    return NormPoint, rate


def nms(bounding_boxes, confidence_score, threshold=myThreshold):
    """
        Non-max Suppression Algorithm
        @param list  Object candidate bounding boxes
        @param list  Confidence score of bounding boxes
        @param float IoU threshold
        @return Rest boxes after nms operation
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score



def getFeatofSource(imgPath: str):
    """
    加载Unet_model，并将imgPath中的图片投入预测模型中，取选定的层进行特征提取，保存到相应的npy文件中
    :param imgPath:
    :return:savePath:返回储存特征矩阵的路径名
    """
    model = UNet(num_kernel=16, kernel_size=3, dim=1, target_dim=1)
    # 将网络拷贝到device中
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # 加载模型参数
    model.load_state_dict(torch.load(r'./UNet_nuclei_trained.pth', map_location=device), False)
    # 测试模式
    model.eval()

    sourceFeat = model.extract_feat_o(imgPath, layer='decode_2')
    sourceFeat_Size = sourceFeat.shape
    print(sourceFeat_Size)

    # 保存特征矩阵,地址为savePath
    # savePath = imgPath.split('/')[-1].split('.')[0] + '_feature'
    savePath = './temporaryFeature'
    np.save(savePath, sourceFeat)
    print("Successfully save!")
    return savePath + '.npy'


def getSliceFeat(featPath=None):
    global NormPoint, sizeRate, normTargetFeatPoint
    NormPoint = global_var.get_value('NormPoint')
    sizeRate = global_var.get_value('rate')
    # NormPoint, sizeRate = selectBoxByMouse(imgMatrix)

    targetFeat = np.load("./temporaryFeature.npy")
    targetFeat_Size = targetFeat.shape[1:][::-1]  # (32, 256, 256)

    FeatWinW, FeatWinH = int(targetFeat_Size[0] * sizeRate[0]), int(targetFeat_Size[1] * sizeRate[-1])

    # 在特征向量层面确定步长,然后再反馈到图像中

    # 获取源的特征向量
    targetFeatPoint = (int(NormPoint[0] * targetFeat_Size[0]), int(NormPoint[1] * targetFeat_Size[1]))
    # 切割的时候是(channel, h, w)
    targetSliceFeat = targetFeat[:, targetFeatPoint[1]:targetFeatPoint[1] + FeatWinH,
                      targetFeatPoint[0]:targetFeatPoint[0] + FeatWinW]
    targetSliceFeat = targetSliceFeat / LA.norm(targetSliceFeat)
    targetSliceFeat = targetSliceFeat.reshape((-1))

    # 进行映射测试
    normTargetFeatPoint = (targetFeatPoint[0] / targetFeat_Size[0], targetFeatPoint[1] / targetFeat_Size[1])

    # 滑动窗口获取全图
    feats = [targetSliceFeat]  # 特征矩阵
    topLeftAxes = [normTargetFeatPoint]  # 每一块左上角坐标,最左上角为(0, 0) ==> (w, h)

    rangeW = targetFeat_Size[0] - FeatWinW + 1
    rangeH = targetFeat_Size[1] - FeatWinH + 1

    for h in range(0, rangeH, stepSizeFeat['stepH']):
        for w in range(0, rangeW, stepSizeFeat['stepW']):
            slice_f = targetFeat[:, h:h + FeatWinH, w:w + FeatWinW]  # 切出的部分# (16, 256, 128)
            norm_feat = slice_f / LA.norm(slice_f)
            # 记录数据
            feats.append(norm_feat.reshape((-1)))
            # 先w后h
            # print(w, h)
            topLeftAxes.append((w / targetFeat_Size[0], h / targetFeat_Size[1]))

    feats = np.array(feats)
    topLeftAxes = np.array(topLeftAxes)

    return targetSliceFeat.reshape((-1)), feats, topLeftAxes


def getRightBoxes(scores_sum, axes, n, scoresThreshold=myThreshold):
    """
    仅用于获取高于阈值的框，并没有做非极大值抑制
    :param scores_sum:
    :param axes:
    :param n:
    :param scoresThreshold:
    :return:
    """
    global sizeRate, winH, winW, imgH, imgW

    scores = scores_sum / n
    # 在原图中画框
    # img = imgMatrix
    # 自定义滑动窗口的大小
    imgH = imgMatrix.shape[0]
    imgW = imgMatrix.shape[1]

    winH = int(imgH * sizeRate[1])
    winW = int(imgW * sizeRate[0])

    bounding_boxes = []
    confidence_score = []
    # 记录高于阈值的框
    for (w, h), s in zip(axes, scores):
        w = int(w * imgW)
        h = int(h * imgH)
        if s >= scoresThreshold:
            bounding_boxes.append((w, h, w + winW, h + winH))
            confidence_score.append(s)
    return bounding_boxes, confidence_score


def matchFeat(sourceVec, feats, topLeftAxes):
    scores_sum = 0
    n = 0
    # 余弦相似度
    scores = np.dot(sourceVec, feats.T)
    scores_sum = np.sum([scores, scores_sum], axis=0)
    n += 1
    bounding_boxes, confidence_score = getRightBoxes(scores_sum, topLeftAxes, n, scoresThreshold=myThreshold)
    return bounding_boxes, confidence_score


def mainNMS(bounding_boxes, confidence_score):
    # Copy image as original
    image = imgMatrix.copy()

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.5  # 字体大小
    thickness = 2  # 字体粗细

    # IoU threshold
    threshold = 0.1

    # draw original box
    print('正常计算的original左上角坐标', (normTargetFeatPoint[0] * imgW, normTargetFeatPoint[1] * imgH))
    cv2.rectangle(image, (int(normTargetFeatPoint[0] * imgW), int(normTargetFeatPoint[1] * imgH)),
                  (int(normTargetFeatPoint[0] * imgW + winW), int(normTargetFeatPoint[1] * imgH + winH)), (0, 0, 255),
                  2)

    # Run non-max suppression algorithm
    picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)
    global_var.set_value('tracked_boxes', picked_boxes)
    # global_var.set_value('score', picked_score)

    # Draw bounding boxes and confidence score after non-maximum supression
    for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
        (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
        cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + 15), (0, 255, 255), -1)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1)
        if confidence > 0.99:
            print('转换后的original左上角坐标', (start_x, start_y))
            cv2.putText(image, 'original', (start_x, start_y + 12), font, font_scale, (255, 0, 0), thickness)
        else:
            cv2.putText(image, str(confidence)[:8], (start_x, start_y + 12), font, font_scale, (0, 0, 0), thickness)

    # Show image
    # global_var.set_value('result_img', image)
    cv2.imwrite('./result_img.jpg', image)  # 正确的路径



def similarTargetDetection(imgPath=None):
    global imgMatrix, myThreshold
    print(imgPath)
    # imgPath = global_var.get_value('firstFrameImgPath')

    # imgPath = r"I:\MySoftware\UI\resource\img.png"
    # savePath = getFeatofSource(imgPath)
    savePath = r'IntegrateApp/temporaryFeature.npy'

    img = cv2.imread(imgPath)
    (width, height) = (512, 512)
    imgMatrix = cv2.resize(img, (width, height), cv2.INTER_LINEAR)

    sourceVec, feats, topLeftAxes = getSliceFeat(savePath)
    bounding_boxes, confidence_score = matchFeat(sourceVec, feats, topLeftAxes)
    myThreshold = 0.75

    # bounding_boxes, confidence_score = nms(bounding_boxes, confidence_score, threshold=0.1)
    mainNMS(bounding_boxes, confidence_score)

    # boxes = [[i[0], i[1], abs(i[0] - i[2]), abs(i[1] - i[3])] for i in bounding_boxes]
    # mainTrack(frame_path, boxes)


if __name__ == '__main__':
    similarTargetDetection(r"I:/Innovation_project/UI/resource/img.png")

