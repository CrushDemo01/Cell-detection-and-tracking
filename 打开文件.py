import os

import cv2
import h5py
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QStyle
from PyQt5.QtGui import QPixmap, QImage, QMovie

from IntegrateApp.多目标跟踪_re import mainTrack, MyTrack
from MyTools import getFeatofSource, similarTargetDetection
# from ui.first1 import My_Ui_MainWindow
from ui.ui2 import My_Ui_MainWindow
from IntegrateApp import global_var
import cgitb

cgitb.enable(format='text')

width = 512
height = 512
global firstImgPath, folderPath


def getInitBox():
    # point,size = (x,y,w,h)
    point = global_var.get_value("point")
    size = global_var.get_value("size")
    print(f"point:{point}\nsize:{size}")
    NormPoint = (point[0] / width, point[1] / height)
    print('左上角坐标比例，区间在(0,1)', NormPoint)
    rate = (size[0] / width, size[1] / height)
    print('高和宽归一化', rate)
    global_var.set_value('NormPoint', NormPoint)
    global_var.set_value('rate', rate)


class Stats():
    def __init__(self):
        super().__init__()
        style = QApplication.style()
        self.width = 512
        self.height = 512
        # self.ui = uic.loadUi("./ui/first.ui")
        self.ui = My_Ui_MainWindow()
        self.ui.run_button.clicked.connect(self.runAll)
        self.ui.track_button.clicked.connect(self.multiTrack)
        self.ui.actionopen.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        self.ui.actionopen.triggered.connect(self.onFileOpen)
        self.ui.actionopen_folder.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        self.ui.actionopen_folder.triggered.connect(self.onFileOpenFolder)

    def onFileOpen(self):
        global firstImgPath
        firstImgPath, _ = QFileDialog.getOpenFileName(self.ui, '选择图片', r'I:\MySoftware\IntegrateApp\UI\resource',
                                              'Image files(*.jpg *.gif *.png)')
        Im = cv2.resize(cv2.imread(firstImgPath), (width, height), cv2.INTER_LINEAR)
        cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)
        qt_img = QImage(Im.data.tobytes(), width, height, QImage.Format_RGB888)
        self.ui.img_label.setPixmap(QPixmap.fromImage(qt_img))
        getFeatofSource(firstImgPath)
        global_var.set_value('firstFrameImgPath', firstImgPath)

    def onFileOpenFolder(self):
        global firstImgPath, folderPath
        folderPath = QFileDialog.getExistingDirectory(self.ui, "选取文件夹", r'I:\MySoftware\IntegrateApp\Fluo-N2DH-GOWT1')  # 起始路径
        # global_var.set_value('folderPath', folderPath)
        print(folderPath)
        filenames = os.listdir(folderPath)
        firstImgPath = os.sep.join([folderPath, filenames[0]])
        Im = cv2.resize(cv2.imread(firstImgPath), (width, height), cv2.INTER_LINEAR)
        cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)
        qt_img = QImage(Im.data.tobytes(), width, height, QImage.Format_RGB888)
        self.ui.img_label.setPixmap(QPixmap.fromImage(qt_img))
        getFeatofSource(firstImgPath)

        # global_var.set_value('firstFrameImgPath', firstImgPath)

    def runAll(self):
        getInitBox()
        similarTargetDetection(firstImgPath)
        self.ui.detected_label.setPixmap(QPixmap('./result_img.jpg'))
        self.ui.detected_label.setScaledContents(True)

    def multiTrack(self):
        global folderPath
        boxes = [[i[0], i[1], abs(i[0] - i[2]), abs(i[1] - i[3])] for i in global_var.get_value('tracked_boxes')]
        # MyTrackFrameList = MyTrack(folderPath, boxes)
        MyTrack(folderPath, boxes)
        # h5f = h5py.File('./TrackFrame.h5', 'r')
        # MyTrackFrameList = h5f['MyTrackFrameList'][:]
        # h5f.close()
        gif = QMovie('MyTrackFrameList.gif')
        self.ui.track_label.setMovie(gif)
        gif.start()

    def multiTrack_(self):
        h5f = h5py.File('./TrackFrame.h5', 'r')
        MyTrackFrameList = h5f['MyTrackFrameList'][:]
        h5f.close()
        for frame in MyTrackFrameList:
            self.ui.track_label.clear()
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(frame.data.tobytes(), width, height, QImage.Format_RGB888)
            self.ui.track_label.setPixmap(QPixmap.fromImage(qt_img))


if __name__ == '__main__':
    app = QApplication([])
    QApplication.processEvents()
    stats = Stats()
    stats.ui.show()
    app.exec_()
