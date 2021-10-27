import cv2
import h5py
import imageio

gif_images = []
h5f = h5py.File('./TrackFrame.h5', 'r')
MyTrackFrameList = h5f['MyTrackFrameList'][:]
h5f.close()
for frame in MyTrackFrameList:
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gif_images.append(frame)

imageio.mimsave("hello.gif", gif_images, fps=5)   # 转化为gif动画

