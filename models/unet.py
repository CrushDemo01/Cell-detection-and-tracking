import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
from .layers import unet_layers as layers
# import .layers.unet_layers as layers
import cv2
from keras.preprocessing import image


class UNet(nn.Module):
    """
    UNet网络结构
    """

    def __init__(self, num_kernel, kernel_size, dim, target_dim):
        """UNet

        Parameters
        ----------
            num_kernel: int
                number of kernels to use for the first layer
            kernel_size: int
                size of the kernel for the first layer
            dims: int
                input data dimention
        """

        super(UNet, self).__init__()

        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.dim = dim
        self.target_dim = target_dim

        # encode
        self.encode_1 = layers.DownSampling(self.dim, num_kernel, kernel_size)
        self.encode_2 = layers.DownSampling(num_kernel, num_kernel * 2, kernel_size)
        self.encode_3 = layers.DownSampling(num_kernel * 2, num_kernel * 4, kernel_size)
        self.encode_4 = layers.DownSampling(num_kernel * 4, num_kernel * 8, kernel_size)

        # bridge
        self.bridge = nn.Conv2d(num_kernel * 8, num_kernel * 16, kernel_size, padding=1, stride=1)

        # decode
        self.decode_4 = layers.UpSampling(num_kernel * 16, num_kernel * 8, kernel_size)
        self.decode_3 = layers.UpSampling(num_kernel * 8, num_kernel * 4, kernel_size)
        self.decode_2 = layers.UpSampling(num_kernel * 4, num_kernel * 2, kernel_size)
        self.decode_1 = layers.UpSampling(num_kernel * 2, num_kernel, kernel_size)

        self.segment = nn.Conv2d(num_kernel, self.target_dim, 1, padding=0, stride=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: Size(1,1,h,w)
        :return:
        以(512,512)为例：
        torch.Size([1, 1, 512, 512])

        torch.Size([1, 16, 256, 256])
        torch.Size([1, 32, 128, 128])
        torch.Size([1, 64, 64, 64])
        torch.Size([1, 128, 32, 32])

        torch.Size([1, 256, 32, 32])

        torch.Size([1, 128, 64, 64])
        torch.Size([1, 64, 128, 128])
        torch.Size([1, 32, 256, 256])
        torch.Size([1, 16, 512, 512])

        torch.Size([1, 1, 512, 512])
        """
        x, skip_1 = self.encode_1(x)
        x, skip_2 = self.encode_2(x)
        x, skip_3 = self.encode_3(x)
        x, skip_4 = self.encode_4(x)

        x = self.bridge(x)

        x = self.decode_4(x, skip_4)
        x = self.decode_3(x, skip_3)
        x = self.decode_2(x, skip_2)
        x = self.decode_1(x, skip_1)

        x = self.segment(x)

        pred = self.activate(x)
        print(pred.shape)

        return pred

    def extract_feat(self, img_path, layer='decode_1', feat_path=None):
        img = image.load_img(img_path, target_size=(512, 512))  # (512, 512)
        # img.size=(224, 224)
        img = image.img_to_array(img)  # img_shape=(224, 224, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)     # ndarray
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, 512, 512)
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中
        x = img_tensor.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          dtype=torch.float32)  # torch.Size([1, 1, 512, 512])

        # 用字典来记录每一层的特征。每一步操作后面记录返回值的size
        features = {'input': x}

        x, skip_1 = self.encode_1(x)  # torch.Size([1, 16, 256, 256])
        features['encode_1'] = x

        x, skip_2 = self.encode_2(x)  # torch.Size([1, 32, 128, 128])
        features['encode_2'] = x

        x, skip_3 = self.encode_3(x)  # torch.Size([1, 64, 64, 64])
        features['encode_3'] = x

        x, skip_4 = self.encode_4(x)  # torch.Size([1, 128, 32, 32])
        features['encode_4'] = x

        x = self.bridge(x)  # torch.Size([1, 256, 32, 32])
        features['bridge'] = x

        x = self.decode_4(x, skip_4)  # torch.Size([1, 128, 64, 64])
        features['decode_4'] = x

        x = self.decode_3(x, skip_3)  # torch.Size([1, 64, 128, 128])
        features['decode_3'] = x

        x = self.decode_2(x, skip_2)  # torch.Size([1, 32, 256, 256])
        features['decode_2'] = x

        x = self.decode_1(x, skip_1)  # torch.Size([1, 16, 512, 512])
        features['decode_1'] = x

        x = self.segment(x)  # torch.Size([1, 1, 512, 512])
        features['segment'] = x

        pred = self.activate(x)
        features['pred'] = pred

        feature = features[layer].cpu().detach().numpy()

        if feat_path is not None:
            # print('x.shape : ', x.shape)
            # print(x.data)     # torch.Size([1, 64, 128, 128]) 矩阵里的值有正有负
            np.save(feat_path, feature)
            # torch.save(x.T, x.shape+"myTensor.pt")

        feature = np.squeeze(feature, axis=0)        # 去掉0维，变成(64, 128, 128)
        norm_feat = feature / LA.norm(feature)
        # print(norm_feat.shape)

        return norm_feat

    def extract_feat_o(self, img_path, layer='decode_1', feat_path=None):
        """
        没有进行归一化处理
        :param img_path:
        :param layer:
        :param feat_path:
        :return:
        """
        img = image.load_img(img_path, target_size=(512, 512))  # (512, 512)
        # img.size=(224, 224)
        img = image.img_to_array(img)  # img_shape=(224, 224, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)     # ndarray
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, 512, 512)
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中
        x = img_tensor.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          dtype=torch.float32)  # torch.Size([1, 1, 512, 512])

        # 用字典来记录每一层的特征。每一步操作后面记录返回值的size
        features = {'input': x}

        x, skip_1 = self.encode_1(x)  # torch.Size([1, 16, 256, 256])
        features['encode_1'] = x

        x, skip_2 = self.encode_2(x)  # torch.Size([1, 32, 128, 128])
        features['encode_2'] = x

        x, skip_3 = self.encode_3(x)  # torch.Size([1, 64, 64, 64])
        features['encode_3'] = x

        x, skip_4 = self.encode_4(x)  # torch.Size([1, 128, 32, 32])
        features['encode_4'] = x

        x = self.bridge(x)  # torch.Size([1, 256, 32, 32])
        features['bridge'] = x

        x = self.decode_4(x, skip_4)  # torch.Size([1, 128, 64, 64])
        features['decode_4'] = x

        x = self.decode_3(x, skip_3)  # torch.Size([1, 64, 128, 128])
        features['decode_3'] = x

        x = self.decode_2(x, skip_2)  # torch.Size([1, 32, 256, 256])
        features['decode_2'] = x

        x = self.decode_1(x, skip_1)  # torch.Size([1, 16, 512, 512])
        features['decode_1'] = x

        x = self.segment(x)  # torch.Size([1, 1, 512, 512])
        features['segment'] = x

        pred = self.activate(x)
        features['pred'] = pred

        feature = features[layer].cpu().detach().numpy()

        if feat_path is not None:
            # print('x.shape : ', x.shape)
            # print(x.data)     # torch.Size([1, 64, 128, 128]) 矩阵里的值有正有负
            np.save(feat_path, feature)
            # torch.save(x.T, x.shape+"myTensor.pt")

        feature = np.squeeze(feature, axis=0)        # 去掉0维，变成(64, 128, 128)
        # norm_feat = feature / LA.norm(feature)
        # print(norm_feat.shape)

        return feature

    def args_dict(self):
        """model arguments to be saved
        """

        model_args = {'dim': self.dim,
                      'target_dim': self.target_dim,
                      'num_kernel': self.num_kernel,
                      'kernel_size': self.kernel_size}

        return model_args

    def display(self):
        print(self)
