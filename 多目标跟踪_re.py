from __future__ import print_function
import h5py
import os
import cv2
from itertools import cycle

import imageio
import numpy as np

from IntegrateApp import global_var
from PyQt5.QtGui import QImage

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def MyTrack(frame_path, bboxes):
    # Set video to load
    filenames = os.listdir(frame_path)
    num = len(filenames)
    (width, height) = (512, 512)
    img_iter = iter(
        [cv2.resize(cv2.imread(os.sep.join([frame_path, x])), (width, height), cv2.INTER_LINEAR) for x in filenames])

    # Read first frame
    frame = next(img_iter)

    # Draw parameters
    color = (0, 255, 255)   # 颜色

    # Specify the tracker type
    trackerType = "CSRT"
    createTrackerByName(trackerType)

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    MyTrackFrameList = []
    # Process video and track objects
    for cnt, frame in enumerate(img_iter):
        success, boxes = multiTracker.update(frame)
        if not success or cnt == num-1:
            break

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, color, 1, 1)

        # save frame
        MyTrackFrameList.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"完成：{cnt}")
    imageio.mimsave("MyTrackFrameList.gif", MyTrackFrameList, fps=5)  # 转化为gif动画


    # f = h5py.File("./TrackFrame.h5", 'w')
    # f.create_dataset('MyTrackFrameList', data=np.array(MyTrackFrameList))
    # f.close()
    # return MyTrackFrameList



def NewTrack(frame_path, bboxes):
    # Set video to load
    filenames = os.listdir(frame_path)
    (width, height) = (512, 512)
    img_iter = iter(
        [cv2.resize(cv2.imread(os.sep.join([frame_path, x])), (width, height), cv2.INTER_LINEAR) for x in filenames])

    # Read first frame
    frame = next(img_iter)

    # Draw parameters
    color = (0, 255, 255)   # 颜色
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects

    # Specify the tracker type
    trackerType = "CSRT"
    createTrackerByName(trackerType)

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    # Process video and track objects
    for cnt, frame in enumerate(img_iter):
        # frame = next(img_iter)
        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if not success:
            break

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)

        # show frame
        cv2.imwrite(f'./result_tracked/res_{cnt}.jpg', frame)  # 正确的路径
        cv2.imshow('MultiTracker', frame)
        i += 1

        # # quit on ESC button
        # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        #     break


def mainTrack(frame_path, bboxes):
    # Set video to load
    filenames = os.listdir(frame_path)
    (width, height) = (512, 512)
    img_iter = cycle(
        [cv2.resize(cv2.imread(os.sep.join([frame_path, x])), (width, height), cv2.INTER_LINEAR) for x in filenames])

    # Read first frame
    frame = next(img_iter)

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.5  # 字体大小
    thickness = 2  # 字体粗细
    color = (0, 255, 255)   # 颜色

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects

    # Specify the tracker type
    trackerType = "CSRT"
    createTrackerByName(trackerType)

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    # Process video and track objects
    while True:
        frame = next(img_iter)
        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if not success:
            break

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break


if __name__ == '__main__':
    # Select boxes
    h5f = h5py.File('boxes&scores.h5', 'r')
    boxes = h5f['boxes'][:]
    # scores = h5f['scores'][:]
    h5f.close()

    print(f'boxes:{boxes}')
    # print(f'scores:{scores}')
    print("Successfully read!")
    frame_path = "./Fluo-N2DH-GOWT1/01"
    boxes = [[i[0], i[1], abs(i[0] - i[2]), abs(i[1] - i[3])] for i in boxes]
    mainTrack(frame_path, boxes)
