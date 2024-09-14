import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import math
import os
import torch.nn as nn
from PIL import Image
from utils.utils import filter2D, USMSharp
from utils.utils_blur import circular_lowpass_kernel, random_mixed_kernels
from utils.utils_resize import random_resizing
from utils.utils_noise import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils.utils_jpeg import DiffJPEG
from utils.matlab_functions import imresize
from torchvision import transforms
from torch.nn import functional as F
from utils.utils import set_seed


class Degradation(nn.Module):
    def __init__(self, scale, gt_size, use_sharp_gt, device):
        super(Degradation, self).__init__()
        ### global settings
        self.scale = scale
        self.gt_size = gt_size
        self.device = device

        ### initization JPEF class
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)

        self.use_sharp_gt = use_sharp_gt
        self.usm_sharpener = USMSharp().to(self.device)
        self.queue_size = 180  # opt.get('queue_size', 180)

        ### the first degradation hypermeters ###
        # 1. blur
        self.blur_kernel_size = 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]  # blur_x / y_sigma
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        # 2. resize
        self.updown_type = ["up", "down", "keep"]
        self.mode_list = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        # 3. noise
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.raw_noise = 5
        # 4. jpeg
        self.jpeg_range = [30, 95]

        ### the second degradation hypermeters ###
        # 1. blur
        self.second_blur_prob = 0.8

        self.blur_kernel_size2 = 21
        self.kernel_range2 = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        # 2. resize
        self.updown_type2 = ["up", "down", "keep"]
        self.mode_list2 = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        # 3. noise
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        # 4. jpeg
        self.jpeg_range2 = [30, 95]

        self.final_sinc_prob = 0.8

        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    @torch.no_grad()
    def forward_deg(self, gt):
        ori_h, ori_w = gt.size()[2:4]
        gt_usm = self.usm_sharpener(gt) if self.use_sharp_gt else gt
        gt_usm_copy = gt_usm.clone()

        # generate kernel
        kernel1 = self.generate_first_kernel().to(self.device)
        kernel2 = self.generate_second_kernel().to(self.device)
        sinc_kernel = self.generate_sinc_kernel().to(self.device)

        # first degradation
        blurred_1 = self.blur_1(gt_usm, kernel1)
        sampled_1 = self.sampled_1(blurred_1)
        noised_1 = self.noise_1(sampled_1)
        compressed_1 = self.jpeg_1(noised_1)

        # second degradation
        blurred_2 = self.blur_2(compressed_1, kernel2)
        sampled_2 = self.sampled_2(blurred_2, ori_h, ori_w)
        noised_2 = self.noise_2(sampled_2)
        compressed_2 = self.jpeg_2(noised_2, ori_h, ori_w, sinc_kernel)

        return compressed_2, gt_usm_copy, kernel1, kernel2, sinc_kernel

    @torch.no_grad()
    def forward(self, gt_path, uint8=False, test=False):
        # read hwc 0-1 numpy
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.

        # augment  !!! if test, please close.
        if not test:
            img_gt = self.augment(img_gt, True, True)

        # numpy 0-1 hwc -> tensor 0-1 chw
        img_gt = self.np2tensor([img_gt], bgr2rgb=False, float32=True, unsqueeze_=True, device=self.device)[0]
        img_gt_copy = img_gt.clone()

        # degradation_piepline
        lq, gt_usm, kernel1, kernel2, sinc_kernel = self.forward_deg(img_gt)

        # clamp and round
        lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.

        # random crop
        # print(f'before crop: gt:{img_gt_copy.shape}, lq:{lq.shape}')
        # paired random crop  !!! if test, please close.
        if not test:
            (gt, gt_usm), lq = self.paired_random_crop([img_gt_copy, gt_usm], lq, self.gt_size, self.scale)
        else:
            gt, gt_usm, lq = img_gt_copy, gt_usm, lq
        # print(f'after crop: gt:{gt.shape}, lq:{lq.shape}')

        if uint8:
            gt, gt_usm, lq = self.tensor2np([gt, gt_usm, lq])
            return gt, gt_usm, lq, kernel1, kernel2, sinc_kernel

        return gt, gt_usm, lq, kernel1, kernel2, sinc_kernel

    @torch.no_grad()
    def forward_interface(self, img_gt, uint8=False):
        """
        img_gt: read gt image, with Tensor [0., 1.] BCHW
        uint8: whether save
        """
        img_gt_copy = img_gt.clone()

        # degradation_piepline
        lq, gt_usm, kernel1, kernel2, sinc_kernel = self.forward_deg(img_gt)

        # clamp and round
        lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.

        gt, gt_usm, lq = img_gt_copy, gt_usm, lq

        if uint8:
            gt, gt_usm, lq = self.tensor2np([gt, gt_usm, lq])

        return gt, gt_usm, lq, kernel1, kernel2, sinc_kernel

    @torch.no_grad()
    def forward_interface_contrast(self, img_gt1, img_gt2, uint8=False, same_deg=False):
        """
        img_gt: read gt image, with Tensor [0., 1.] BCHW
        uint8: whether save
        """
        img_gt_copy1 = img_gt1.clone()
        img_gt_copy2 = img_gt2.clone()

        # degradation_piepline
        if same_deg:
            randseed = random.randint(0, 100000000)
            set_seed(randseed)
            lq1, gt_usm1, kernel11, kernel21, sinc_kernel1 = self.forward_deg(img_gt1)
            set_seed(randseed)
            lq2, gt_usm2, kernel12, kernel22, sinc_kernel2 = self.forward_deg(img_gt2)
        else:
            lq1, gt_usm1, kernel11, kernel21, sinc_kernel1 = self.forward_deg(img_gt1)
            lq2, gt_usm2, kernel12, kernel22, sinc_kernel2 = self.forward_deg(img_gt2)

        # clamp and round
        lq1 = torch.clamp((lq1 * 255.0).round(), 0, 255) / 255.
        lq2 = torch.clamp((lq2 * 255.0).round(), 0, 255) / 255.

        gt1, gt_usm1, lq1 = img_gt_copy1, gt_usm1, lq1
        gt2, gt_usm2, lq2 = img_gt_copy2, gt_usm2, lq2

        if uint8:
            gt1, gt_usm1, lq1 = self.tensor2np([gt1, gt_usm1, lq1])
            gt2, gt_usm2, lq2 = self.tensor2np([gt2, gt_usm2, lq2])

        return gt1, gt_usm1, lq1, gt2, gt_usm2, lq2

    def blur_1(self, img, kernel1):
        img = filter2D(img, kernel1)
        return img

    def blur_2(self, img, kernel2):
        if np.random.uniform() < self.second_blur_prob:
            img = filter2D(img, kernel2)
        return img

    def sampled_1(self, img):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(img, scale_factor=scale, mode=mode)

        return img

    def sampled_2(self, img, ori_h, ori_w):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(
            img, size=(int(ori_h / self.scale * scale), int(ori_w / scale * scale)), mode=mode)

        return img

    def noise_1(self, img):
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob:
            img = random_add_gaussian_noise_pt(img, sigma_range=self.noise_range, clip=True, rounds=False,
                                               gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def noise_2(self, img):
        gray_noise_prob = self.gray_noise_prob2
        if np.random.uniform() < self.gaussian_noise_prob2:
            img = random_add_gaussian_noise_pt(
                img, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def jpeg_1(self, img):
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*self.jpeg_range)
        img = torch.clamp(img, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        img = self.jpeger(img, quality=jpeg_p)
        return img

    def jpeg_2(self, out, ori_h, ori_w, sinc_kernel):
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)

        return out

    def isp(self, img):
        # tensor 0-1 bchw -> numpy 0-1 hwc
        img = img.numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
        clean_img, noise_img, sigma_img = self.isper.noise_generate_srgb(img)
        noise_img = torch.from_numpy(noise_img.transpose(2, 0, 1)).float().unsqueeze(0)
        return noise_img

    def isp_batch(self, imgs):
        out_isp = torch.zeros(imgs.shape, device=imgs.device, dtype=imgs.dtype)
        imgs = imgs.data.cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
        for b in range(imgs.shape[0]):
            _, noise, _ = self.isper.noise_generate_srgb(imgs[b, ...])
            out_isp[b, ...] = torch.from_numpy(noise.transpose(2, 0, 1)).float()
        # out = out_isp.clone()
        # del out_isp

        return out_isp

    def generate_first_kernel(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel)

    def generate_second_kernel(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel2)

    def generate_sinc_kernel(self):
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        return sinc_kernel

    def np2tensor(self, imgs, bgr2rgb=False, float32=True, unsqueeze_=False, device='cuda'):
        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            if unsqueeze_:
                img = img.unsqueeze(0)
            return img.to(device)

        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        else:
            return _totensor(imgs, bgr2rgb, float32)

    def tensor2np(self, imgs):
        def _tonumpy(img):
            img = img.data.cpu().numpy().squeeze(0).transpose(1, 2, 0)  # .astype(np.float32)
            img = np.uint8((img.clip(0, 1) * 255.).round())
            return img

        if isinstance(imgs, list):
            return [_tonumpy(img) for img in imgs]
        else:
            return _tonumpy(imgs)

    def save_imgs(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        mul_value = 1. if max(imgs[0]) > 128. else 255.

        if type(imgs[0]) == torch.Tensor:
            imgs = self.tensor2np(imgs)
            pass

        elif type(imgs[0]) == np.ndarray:
            pass

    def augment(self, imgs, hflip=True, rotation=True, flows=None, return_status=False):
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        return imgs

    def paired_random_crop(self, img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
        """Paired random crop. Support Numpy array and Tensor inputs.

        It crops lists of lq and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            gt_patch_size (int): GT patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth. Default: None.

        Returns:
            list[ndarray] | ndarray: GT images and LQ images. If returned results
                only have one element, just return ndarray.
        """

        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]

        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

        if input_type == 'Tensor':
            h_lq, w_lq = img_lqs[0].size()[-2:]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            h_lq, w_lq = img_lqs[0].shape[0:2]
            h_gt, w_gt = img_gts[0].shape[0:2]
        lq_patch_size = gt_patch_size // scale

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                             f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                             f'({lq_patch_size}, {lq_patch_size}). '
                             f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch
        if input_type == 'Tensor':
            img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
        else:
            img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        if input_type == 'Tensor':
            img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        else:
            img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        return img_gts, img_lqs


if __name__ == '__main__':
    from utils.utils import _get_paths_from_images
    from utils.utils import uint2tensor

    # print(os.path.abspath(os.path.join(__file__, os.path.pardir)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deg_pipeline = Degradation(scale=2, gt_size=480, use_sharp_gt=True, device=device)

    """ Demo-1: Generate degraded images and save them. """

    # input path: directory or single-image path
    input_path = r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png'
    save_path = r'figs'
    os.makedirs(save_path, exist_ok=True)

    if os.path.isdir(input_path):
        gt_paths = _get_paths_from_images(input_path)
    else:
        gt_paths = [input_path]

    for gt_path in gt_paths:
        base, ext = os.path.splitext(os.path.basename(gt_path))

        gt, gt_usm, lq, kernel1, kernel2, sinc_kernel = deg_pipeline(gt_path, uint8=True, test=True)

        cv2.imwrite(os.path.join(save_path, f'{base}_deg{ext}'), lq[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, f'{base}_gt{ext}'), gt[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, f'{base}_gtusm{ext}'), gt_usm[:, :, ::-1])

    """ Demo-2: Accessing the Dataset class via interface forward_interface. """

    gt, gt_usm, lq, kernel1, kernel2, sinc_kernel = deg_pipeline.forward_interface(
        uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device)
    )

    """ Demo-3: Accessing the Dataset class via interface forward_interface_contrast,
                achieving the same degradation of two image patches. (For some contrast learning based methods). """

    gt1, gt_usm1, lq1, gt2, gt_usm2, lq2 = deg_pipeline.forward_interface_contrast(
        uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device),
        uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device),
        same_deg=True
    )
    print((lq1 == lq2).all())
    # tensor(True, device='cuda:0')
