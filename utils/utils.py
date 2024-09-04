import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import os
from natsort import os_sorted
import torch.nn as nn
import random

"""
filter2D(img, kernel)
usm_sharp(img, weight=0.5, radius=50, threshold=10)
USMSharp(torch.nn.Module)
"""


def uint2tensor(img, device='cpu'):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0).to(device)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def single2uint8(img):
    return np.uint8((img.clip(0., 1.) * 255).round())


def uint2single(img):
    return img.astype(np.float32) / 255.


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        # raise ValueError('Wrong kernel size')
        img = F.pad(img, (k // 2, k // 2 - 1, k // 2, k // 2 - 1), mode='reflect')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            self.pad = nn.ReflectionPad2d(
                (kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B * C).view((B, C, H, W))


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.
    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):
    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def savekernel(mat_file, save_path):
    from scipy.io import loadmat, savemat
    kernel = loadmat(mat_file)['Kernel']
    k = kernel - np.min(kernel)
    k = k / np.max(k)
    plt.imshow(k * 255)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(save_path, transparent=True)
    plt.show()
    # plt.imshow(features_struct)
    # plt.tight_layout()
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    img = cv2.imread('../dog.jpg')
    img = usm_sharp(img)
    cv2.imwrite('../usm.jpg', img)
    # img=np.uint8((img.clip(0,1) * 255.).round())
    #
    # img=img[:,:,::-1]
    # plt.imshow(img)
    # plt.show()
