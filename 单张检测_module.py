import os
from itertools import cycle

import cv2
# import h5py
import numpy as np
import torch
from numpy import linalg as LA

from models import UNet
# from 鼠标框选 import selectBoxByMouse
from 多目标跟踪_re import mainTrack
from IntegrateApp import global_var

global NormPoint, sizeRate, winH, winW, imgH, imgW, normTargetFeatPoint
global imgMatrix
global stepSizeFeat
stepSizeFeat = {
    'stepH': 4,
    'stepW': 4
}


def selectBoxByMouse(img):
    (min_x, min_y, width, height) = cv2.selectROI('MultiTracker', img)
    print('左上角坐标', (min_x, min_y))
    NormPoint = (min_x / img.shape[1], min_y / img.shape[0])
    print('左上角坐标比例，区间在(0,1)', NormPoint)
    print('高和宽', (height, width))
    rate = (width / img.shape[1], height / img.shape[0])
    print('高和宽归一化', rate)
    return NormPoint, rate


def nms(bounding_boxes, confidence_score, threshold):
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
    model.load_state_dict(torch.load('UNet_nuclei_trained.pth', map_location=device), False)
    # 测试模式
    model.eval()

    sourceFeat = model.extract_feat_o(imgPath, layer='decode_2')
    sourceFeat_Size = sourceFeat.shape
    print(sourceFeat_Size)

    # 保存特征矩阵,地址为savePath
    savePath = './temporaryFeature'
    np.save(savePath, sourceFeat)
    print("Successfully save!")
    return savePath + '.npy'



def getMainTargetFeat(featPath):
    """
    得到每次选取正向目标的一维特征，左上角点的相对坐标，宽和高的相对大小
    """
    global sizeRate
    # norm_Point, size_Rate = selectBoxByMouse(imgMatrix)
    norm_Point = (0.35546875, 0.263671875)
    size_Rate = (0.0703125, 0.0703125)

    sizeRate = size_Rate
    targetFeat = np.load(featPath)
    targetFeat_Size = targetFeat.shape[1:][::-1]  # (32, 256, 256)
    FeatWinW, FeatWinH = int(targetFeat_Size[0] * size_Rate[0]), int(targetFeat_Size[1] * size_Rate[-1])
    # 获取源的特征向量
    targetFeatPoint = (int(norm_Point[0] * targetFeat_Size[0]), int(norm_Point[1] * targetFeat_Size[1]))
    # 切割的时候是(channel, h, w)
    targetSliceFeat = targetFeat[:, targetFeatPoint[1]:targetFeatPoint[1] + FeatWinH,
                      targetFeatPoint[0]:targetFeatPoint[0] + FeatWinW]
    targetSliceFeat = targetSliceFeat / LA.norm(targetSliceFeat)    # 归一化
    targetSliceFeat = targetSliceFeat.reshape((-1))
    return targetSliceFeat.reshape((-1)), norm_Point, size_Rate


def getSecondTargetFeat(featPath, size_Rate):
    """
    得到每次选取正向目标的一维特征，左上角点的相对坐标
    宽和高的相对大小用第一次的
    """
    norm_Point, _ = selectBoxByMouse(imgMatrix)
    # norm_Point = (0.779296875, 0.142578125)
    targetFeat = np.load(featPath)
    targetFeat_Size = targetFeat.shape[1:][::-1]  # (32, 256, 256)
    FeatWinW, FeatWinH = int(targetFeat_Size[0] * sizeRate[0]), int(targetFeat_Size[1] * sizeRate[-1])
    # 获取源的特征向量
    targetFeatPoint = (int(norm_Point[0] * targetFeat_Size[0]), int(norm_Point[1] * targetFeat_Size[1]))
    # 切割的时候是(channel, h, w)
    targetSliceFeat = targetFeat[:, targetFeatPoint[1]:targetFeatPoint[1] + FeatWinH,
                      targetFeatPoint[0]:targetFeatPoint[0] + FeatWinW]
    targetSliceFeat = targetSliceFeat / LA.norm(targetSliceFeat)    # 归一化
    targetSliceFeat = targetSliceFeat.reshape((-1))
    return targetSliceFeat.reshape((-1))


def getSliceFeat(featPath, size_Rate):
    # 读取文件，获取特征向量
    targetFeat = np.load(featPath)
    targetFeat_Size = targetFeat.shape[1:][::-1]  # (32, 256, 256)
    # 获得特征窗口的大小
    FeatWinW, FeatWinH = int(targetFeat_Size[0] * size_Rate[0]), int(targetFeat_Size[1] * size_Rate[-1])

    # 滑动窗口获取全图
    feats = []  # 特征矩阵
    topLeftAxes = []  # 每一块左上角坐标,最左上角为(0, 0) ==> (w, h)

    rangeW = targetFeat_Size[0] - FeatWinW + 1
    rangeH = targetFeat_Size[1] - FeatWinH + 1

    for h in range(0, rangeH, stepSizeFeat['stepH']):
        for w in range(0, rangeW, stepSizeFeat['stepW']):
            slice_f = targetFeat[:, h:h + FeatWinH, w:w + FeatWinW]  # 切出的部分# (16, 256, 128)
            norm_feat = slice_f / LA.norm(slice_f)
            # 记录数据
            feats.append(norm_feat.reshape((-1)))
            # 先w后h
            topLeftAxes.append((w / targetFeat_Size[0], h / targetFeat_Size[1]))

    feats = np.array(feats)
    topLeftAxes = np.array(topLeftAxes)

    return feats, topLeftAxes


def matchFeat(sourceVec, feats, topLeftAxes, threshold):
    global sizeRate, winH, winW, imgH, imgW
    # 余弦相似度
    scores = np.dot(sourceVec, feats.T)
    # 自定义滑动窗口的大小
    imgH = imgMatrix.shape[0]
    imgW = imgMatrix.shape[1]

    winH = int(imgH * sizeRate[1])
    winW = int(imgW * sizeRate[0])

    bounding_boxes = []
    confidence_score = []
    newAxes = []
    newFeat = []
    # 记录高于阈值的框
    i = 0
    for (w, h), s in zip(topLeftAxes, scores):
        wPx = int(w * imgW)
        hPx = int(h * imgH)
        if s >= threshold:
            bounding_boxes.append((wPx, hPx, wPx + winW, hPx + winH))
            confidence_score.append(s)
            newAxes.append((w, h))
            newFeat.append(feats[i])
        i += 1

    return bounding_boxes, confidence_score, newAxes, np.array(newFeat)


def secondMatch(sourceVec, feats, topLeftAxes, threshold):
    global sizeRate, winH, winW, imgH, imgW
    # 余弦相似度
    scores = np.dot(sourceVec, feats.T)
    # 自定义滑动窗口的大小
    imgH = imgMatrix.shape[0]
    imgW = imgMatrix.shape[1]

    winH = int(imgH * sizeRate[1])
    winW = int(imgW * sizeRate[0])

    bounding_boxes = []
    confidence_score = []
    newAxes = []
    newFeat = []
    # 记录高于阈值的框
    i = 0
    for (w, h), s in zip(topLeftAxes, scores):
        wPx = int(w * imgW)
        hPx = int(h * imgH)
        if s >= threshold:
            bounding_boxes.append((wPx, hPx, wPx + winW, hPx + winH))
            confidence_score.append(s)
            newAxes.append((w, h))
            newFeat.append(feats[i])
        i += 1
    return bounding_boxes, confidence_score, newAxes, np.array(newFeat)


def mainNMS(bounding_boxes, confidence_score):
    # Copy image as original
    image = imgMatrix.copy()

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.5  # 字体大小
    thickness = 2  # 字体粗细

    # IoU threshold
    threshold = 0.1

    # Run non-max suppression algorithm
    picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

    # Draw bounding boxes and confidence score after non-maximum supression
    for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1)
        cv2.imshow('NMS', image)
        cv2.waitKey(1)

    # Show image
    cv2.imshow('NMS', image)
    cv2.imwrite('./result_img.jpg', image)  # 正确的路径
    cv2.waitKey()
    cv2.destroyAllWindows()  # 最后释放窗口
    # boxes = [[i[0], i[1], abs(i[0] - i[2]), abs(i[1] - i[3])] for i in picked_boxes]
    # mainTrack(frame_path, boxes)
    return picked_boxes, picked_score


def similarTargetDetection(imagesPath=None):
    global imgMatrix, myThreshold
    # frame_path = "./Fluo-N2DH-GOWT1/01"
    # # imgPath = r'./Fluo-N2DH-GOWT1/01/t000.tif'
    # filenames = os.listdir(frame_path)
    # fristFramePath = os.sep.join([frame_path, filenames[0]])
    fristFramePath = r"./resource/t000.tif"

    savePath = getFeatofSource(imgPath=fristFramePath)
    # savePath = r't000_feature.npy'

    img = cv2.imread(fristFramePath)
    (width, height) = (512, 512)
    imgMatrix = cv2.resize(img, (width, height), cv2.INTER_LINEAR)
    sourceVec, norm_Point, size_Rate = getMainTargetFeat(savePath)
    feats, topLeftAxes = getSliceFeat(savePath, size_Rate)
    # Initial threshold
    myThreshold = 0.7
    bounding_boxes, confidence_score, newAxes, newFeat = matchFeat(sourceVec, feats, topLeftAxes, myThreshold)
    bounding_boxes, confidence_score = mainNMS(bounding_boxes, confidence_score)
    # k = input("要继续吗？[Y]/N: ")
    # if k == '' or k == 'Y' or k == 'y':
    #     # myThreshold += 0.1
    #
    #     while True:
    #         sourceVec = getSecondTargetFeat(savePath, size_Rate)
    #         bounding_boxes, confidence_score, newAxes, newFeat = secondMatch(sourceVec, newFeat, newAxes, myThreshold)
    #         # bounding_boxes, confidence_score = nms(bounding_boxes, confidence_score, threshold=0.1)
    #         mainNMS(bounding_boxes, confidence_score)
    #
    #         k = input("要继续吗？[Y]/N: ")
    #         if k == '' or k == 'Y' or k == 'y':
    #             myThreshold += 0.1
    #             continue
    #         elif k == 'N' or k == 'n':
    #             break
    #         else:
    #             print("输入错误！")
    #             break
    #
    # elif k == 'N' or k == 'n':
    #     pass
    # else:
    #     print("输入错误！")
    #     pass

    # boxes = [[i[0], i[1], abs(i[0] - i[2]), abs(i[1] - i[3])] for i in bounding_boxes]
    # mainTrack(frame_path, boxes)
    while True:
        sourceVec = getSecondTargetFeat(savePath, size_Rate)
        bounding_boxes, confidence_score, newAxes, newFeat = secondMatch(sourceVec, newFeat, newAxes, myThreshold)
        # bounding_boxes, confidence_score = nms(bounding_boxes, confidence_score, threshold=0.1)
        mainNMS(bounding_boxes, confidence_score)

"""
先将阈值设到最低，
"""

if __name__ == '__main__':
    # frame_path = "./Fluo-N2DH-GOWT1/01"
    # similarTargetDetection(frame_path)
    similarTargetDetection()
    # p = "../UI/resource/img.png"
    # (width, height) = (512, 512)
    # selectBoxByMouse(cv2.resize(cv2.imread(p), (width, height), cv2.INTER_LINEAR))

