import torch.utils.data as data
import torch
import PIL.Image as Image
# from data_process import *
import os
import random
from math import ceil
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
import cv2
import math
from glob import glob
from tqdm import tqdm,trange
import random
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io,transform
import imageio
from torchvision.transforms import Compose
import platform
sysstr = platform.system()
import warnings
import h5py

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
def arugment(img,gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    if hflip:
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip:
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, gt
def augment_img_3(img: np.ndarray, mode: int = 0) -> np.ndarray:
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        raise ValueError

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]
def get_patch_3(img, gt, lq, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :],lq[ty:ty + tp, tx:tx + tp, :]
def train_transform(degree=180):

    return transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=degree),
        transforms.ColorJitter(),
    ])
def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x
def tensor2uint(img,times=255):
    img = img.data.float().cpu().numpy()
    img = np.transpose(img,(1,2,0))
    return img*times
def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # print('0',img.shape)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            # print('1',img.shape)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,:1]
            # print('2',img.shape)
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img
    else:
        raise NotImplementedError
    return img

class Dataset_RGBDD(data.Dataset):
    def __init__(self, path, syn,patch=64,aug_mode=True, scale=4,train = True):
        super(Dataset_RGBDD, self).__init__()
        self.scale = scale
        self.syn = syn
        self.train = train
        self.patch = patch
        self.aug_mode = aug_mode
        types = ['models', 'plants', 'portraits']
        self.HQ_list = []
        self.Guide_list = []
        self.LQ_list = []
        for type in types:
            if self.train:
                list_dir = os.listdir('%s/%s/%s_train' % (path, type, type))
                for n in list_dir:
                    self.Guide_list.append('%s/%s/%s_train/%s/%s_RGB.jpg' % (path, type, type, n, n))
                    self.HQ_list.append('%s/%s/%s_train/%s/%s_HR_gt.png' % (path, type, type, n, n))
                    self.LQ_list.append('%s/%s/%s_train/%s/%s_LR_fill_depth.png' % (path, type, type, n, n))
            else:
                list_dir = os.listdir('%s/%s/%s_test' % (path, type, type))
                for n in list_dir:
                    self.Guide_list.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (path, type, type, n, n))
                    self.HQ_list.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (path, type, type, n, n))
                    self.LQ_list.append('%s/%s/%s_test/%s/%s_LR_fill_depth.png' % (path, type, type, n, n))
    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor
    def load_depth(self, depth_data):
        return (np.array(Image.open(depth_data))).astype(np.float32)
    def load_color(self, color_data):
        return (np.array(Image.open(color_data))).astype(np.float32)
    def load_color_y(self, color_data):
        color_data = ((np.array(Image.open(color_data).convert('YCbCr'))[:, :, 0])).astype(np.float32)
        return color_data
    def argument(self, hq,lq,guide):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
            guide = torch.flip(guide, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
            guide = torch.flip(guide, dims=[1]).clone()
        return hq, lq, guide
    def get_patch_random(self, hq, lq, guide):
        win = self.patch
        h, w = hq.shape[1:3]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        hq = hq[:,y:y + win, x:x + win]
        lq = lq[:,y:y + win, x:x + win]
        guide = guide[:,y:y + win, x:x + win]
        return hq,lq,guide
    def __len__(self):
        return len(self.Guide_list)
    def __getitem__(self, idx):
        Guide_data = self.load_color_y(self.Guide_list[idx])
        # Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_depth(self.HQ_list[idx])

        h, w = HQ_data.shape[:2]
        if self.syn:
            LQ_data = np.array(Image.fromarray(HQ_data).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        else:
            LQ_data = np.array(Image.open(self.LQ_list[idx]).resize((w // self.scale, h // self.scale), Image.BICUBIC)).astype(np.float32)
        LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))

        maxx = np.max(HQ_data)
        minn = np.min(HQ_data)
        LQ_data = (LQ_data - minn) / (maxx - minn)
        guide_max = np.max(Guide_data)
        guide_min = np.min(Guide_data)
        Guide_data = (Guide_data - guide_min) / (guide_max - guide_min)
        HQ_maxx = np.max(HQ_data)
        HQ_minn = np.min(HQ_data)
        HQ_data = (HQ_data - HQ_minn) / (HQ_maxx - HQ_minn)
        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = np.expand_dims(LQ_data, axis=-1)
        Guide_data = np.expand_dims(Guide_data, axis=-1)

        LQ_data = self.totensor(LQ_data)
        Guide_data = self.totensor(Guide_data)
        HQ_data = self.totensor(HQ_data)
        if self.train:
            if self.aug_mode:
                HQ_data,LQ_data, Guide_data = self.argument(HQ_data, LQ_data,Guide_data)
            HQ_data,LQ_data, Guide_data = self.get_patch_random(HQ_data, LQ_data, Guide_data)
        # a = tensor2uint(HQ_data.detach().float())
        # b = tensor2uint(LQ_data.detach().float())
        # c = tensor2uint(Guide_data.detach().float())
        # cv2.imwrite('patch1.png', a)
        # cv2.imwrite('patch2.png', b)
        # cv2.imwrite('patch3.png', c)
        return LQ_data, Guide_data,HQ_data
class Dataset_Middle_LU(data.Dataset):
    def __init__(self,path,scale):
        super(Dataset_Middle_LU, self).__init__()
        self.scale = scale
        HQ_dir = os.path.join(path, "depth")
        Guide_dir = os.path.join(path, "rgb")
        files = os.listdir(HQ_dir)
        files.sort()
        self.HQ_list = [os.path.join(HQ_dir, file) for file in files]
        files = os.listdir(Guide_dir)
        files.sort()
        self.Guide_list = [os.path.join(Guide_dir, file) for file in files]
    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor
    def modcrop(self,img):
        h, w = img.shape[0], img.shape[1]
        h = h - h % self.scale
        w = w - w % self.scale
        return img[:h, :w]
    def load_depth(self, depth_data):
        depth_data = (np.array(Image.open(depth_data)) / 255.).astype(np.float32)
        return self.modcrop(depth_data)
    def load_color(self, color_data):
        color_data = (np.array(Image.open(color_data)) / 255. ).astype(np.float32)
        return self.modcrop(color_data)
    def load_color_y(self, color_data):
        color_data = (np.array(Image.open(color_data).convert('YCbCr'))[:, :, 0] / 255. ).astype(np.float32)
        return self.modcrop(color_data)
    def __len__(self):
        return len(self.Guide_list)
    def __getitem__(self, idx):
        # Guide_data = self.load_color_y(self.Guide_list[idx])
        Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_depth(self.HQ_list[idx])
        h, w = HQ_data.shape[:2]
        # print(HQ_data.shape,np.max(HQ_data))
        LQ_data = np.array(Image.fromarray(HQ_data).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        # LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))
        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = np.expand_dims(LQ_data, axis=-1)
        # Guide_data = np.expand_dims(Guide_data, axis=-1)

        # print(LQ_data.shape,Guide_data.shape,HQ_data.shape)
        # Guide_data = cv2.cvtColor(Guide_data, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('patch1.png', LQ_data*255)
        # cv2.imwrite('patch2.png', HQ_data*255)
        # cv2.imwrite('patch3.png', Guide_data*255)

        return self.totensor(LQ_data),self.totensor(Guide_data),self.totensor(HQ_data)
class NYU_v2_datset(data.Dataset):
    def __init__(self, root_dir, patch=64, scale=4, train=True):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.scale = scale
        self.train = train
        self.patch = patch
        if train:
            self.depths = np.load('%s/train_depth_split.npy' % root_dir)
            self.images = np.load('%s/train_images_split.npy' % root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy' % root_dir)
            self.images = np.load('%s/test_images_v2.npy' % root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def totensor_guide(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor
    def tensor_Y_tensor(self,data):
        data  = tensor2uint(data.detach().float(),times=1)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
        data = np.expand_dims(data[:, :, 0], axis=-1)
        return self.totensor_guide(data)
    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]

        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth, 2), patch_size=self.patch)
            image, depth = arugment(img=image, gt=depth)
        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w // s, h // s), Image.BICUBIC))
        bicubic = np.array(Image.fromarray(lr).resize((w, h), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            # lr = self.transform(np.expand_dims(lr, 2)).float()
            bicubic = self.transform(np.expand_dims(bicubic, 2)).float()
        image = self.tensor_Y_tensor(image)
        # print(image.shape)
        return bicubic, image, depth
class Dataset_CIDIS_c3(data.Dataset):
    def __init__(self,dataset,path,patch=64,scale=4,train=True):
        super(Dataset_CIDIS_c3, self).__init__()
        self.scale = scale
        self.patch = patch
        self.train = train
        self.dataset = dataset
        if self.scale == 4:
            self.syn_LR = True
        else:
            self.syn_LR = False
        self.transform = transforms.Compose([transforms.ToTensor()])
        if dataset=='CIDIS':
            HQ_dir = os.path.join(path, "thermal")
            Guide_dir = os.path.join(path, "visible")
        elif dataset=='CIDIS_200' or dataset=='CIDIS_200_enhance':
            HQ_dir = os.path.join(path, "thermal")
            Guide_dir = os.path.join(path, "visible")
            if not self.syn_LR:
                LQ_dir = os.path.join(path, "thermal_x"+str(scale))
        elif dataset=='LLVIP':
            HQ_dir = os.path.join(path, "infrared")
            Guide_dir = os.path.join(path, "visible")
        elif dataset=='MSRS':
            HQ_dir = os.path.join(path, "ir")
            Guide_dir = os.path.join(path, "vi")
        elif dataset=='ULB17':
            HQ_dir = os.path.join(path, "HR_Thermal")
            Guide_dir = os.path.join(path, "LR_Thermal")
        files = os.listdir(HQ_dir)
        files.sort()
        self.HQ_list = [os.path.join(HQ_dir, file) for file in files]
        files = os.listdir(Guide_dir)
        files.sort()
        self.Guide_list = [os.path.join(Guide_dir, file) for file in files]
        if dataset=='CIDIS_200' or dataset=='CIDIS_200_enhance':
            if not self.syn_LR:
                files = os.listdir(LQ_dir)
                files.sort()
                self.LQ_list = [os.path.join(LQ_dir, file) for file in files]
        # print(len(self.LQ_list),len(self.HQ_list))
    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor
    def modcrop(self,img):
        h, w = img.shape[0], img.shape[1]
        h = h - h % self.scale
        w = w - w % self.scale
        return img[:h, :w]
    def load_depth(self, depth_data):
        depth_data = (np.array(Image.open(depth_data)) / 255.).astype(np.float32)
        return self.modcrop(depth_data)
    def load_thermal(self, depth_data):
        data = np.array(Image.open(depth_data))
        # print(data.shape)
        if len(data.shape)==2:
            data = np.expand_dims(data, axis=-1)
        depth_data = (data[:, :, 0:1] / 255.).astype(np.float32)
        return depth_data
    def load_thermal_lq(self, depth_data):
        data = np.array(Image.open(depth_data))
        # print(data.shape)
        if len(data.shape)==2:
            data = np.expand_dims(data, axis=-1)
        depth_data = (data[:, :, 0:1] / 255.).astype(np.float32)
        return depth_data
    def load_color(self, color_data):
        color_data = (np.array(Image.open(color_data)) / 255. ).astype(np.float32)
        return self.modcrop(color_data)
    def load_color_y(self, color_data):
        color_data = (np.array(Image.open(color_data).convert('YCbCr'))[:, :, 0:1] / 255. ).astype(np.float32)
        return self.modcrop(color_data)
    def __len__(self):
        return len(self.Guide_list)
    def __getitem__(self, idx):
        Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_thermal(self.HQ_list[idx])
        if self.dataset=='CIDIS_200' or self.dataset=='CIDIS_200_enhance' and not self.syn_LR:
            LQ_data = self.load_thermal_lq(self.LQ_list[idx])
            h, w = HQ_data.shape[:2]
            s = self.scale
            bicubic = np.array(Image.fromarray(LQ_data.squeeze()).resize((w, h), Image.BICUBIC))
            bicubic = np.expand_dims(bicubic, 2)
            # print(Guide_data.shape,HQ_data.shape,bicubic.shape)
            if self.train:
                Guide_data, HQ_data, bicubic = get_patch_3(img=Guide_data, gt=HQ_data,lq=bicubic, patch_size=self.patch)
                mode=np.random.randint(0, 8)
                Guide_data = augment_img_3(Guide_data,mode)
                HQ_data = augment_img_3(HQ_data,mode)
                bicubic = augment_img_3(bicubic,mode)
        else:
            if self.train:
                Guide_data, HQ_data = get_patch(img=Guide_data, gt=HQ_data, patch_size=self.patch)
                mode = np.random.randint(0, 8)
                Guide_data = augment_img_3(Guide_data, mode)
                HQ_data = augment_img_3(HQ_data, mode)
            h, w = HQ_data.shape[:2]
            s = self.scale
            lr = np.array(Image.fromarray(HQ_data.squeeze()).resize((w // s, h // s), Image.BICUBIC))
            bicubic = np.array(Image.fromarray(lr).resize((w, h), Image.BICUBIC))
            bicubic = np.expand_dims(bicubic, 2)

        # print(bicubic.shape, Guide_data.shape, HQ_data.shape)
        # cv2.imwrite('patch1.png', bicubic * 255)
        # cv2.imwrite('patch2.png', HQ_data * 255)
        # cv2.imwrite('patch3.png', Guide_data * 255)
        return self.totensor(bicubic), self.totensor(Guide_data), self.totensor(HQ_data)
class Dataset_pansharpen(data.Dataset):
    def __init__(self,path,img_scale,train):
        super(Dataset_pansharpen, self).__init__()
        self.img_scale = img_scale
        if train:
            data = h5py.File(path)
            self.process_ms(data)
            self.process_lms(data)
            self.process_pan(data)
            self.process_gt(data)

        else:
            data = h5py.File(path)
            self.process_ms(data)
            self.process_lms(data)
            self.process_pan(data)
            self.process_gt(data)

    def process_gt(self, data):
        if data.get('gt', None) is None:
            self.gt = self.lms
        else:
            gt = data["gt"][...]  # convert to np tpye for CV2.filter
            gt = np.array(gt, dtype=np.float32) / self.img_scale
            self.gt = torch.from_numpy(gt)  # NxCxHxW:
        # print('gt', self.gt.shape)

    def process_lms(self, data):
        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        lms = np.array(lms, dtype=np.float32) / self.img_scale
        self.lms = torch.from_numpy(lms)
        # print('lms',self.lms.shape)


    def process_ms(self, data):
        ms = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms = np.array(ms, dtype=np.float32) / self.img_scale
        self.ms = torch.from_numpy(ms) # NxCxHxW:
        # print('ms',self.ms.shape)

    def process_pan(self, data):
        pan = data['pan'][...]  # Nx1xHxW
        pan = np.array(pan, dtype=np.float32) / self.img_scale
        self.pan = torch.from_numpy(pan)
        # print('pan', self.lms.shape)
    def __getitem__(self, index):

        return {'gt': self.gt[index, :, :, :],
                'lms': self.lms[index, :, :, :],
                'ms': self.ms[index, :, :, :],
                'pan': self.pan[index, ...]}

    def __len__(self):
        return self.gt.shape[0]


def save_tensor_channels(tensor, output_dir='',name='gt'):
    os.makedirs(output_dir, exist_ok=True)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("输入张量形状必须为[1, C, H, W]")
    tensor = tensor.squeeze(0)
    num_channels = tensor.shape[0]

    for c in range(num_channels):
        channel = tensor[c, :, :]
        channel_np = channel.cpu().detach().numpy()
        if channel_np.max() > 1.0:
            channel_np = (channel_np - channel_np.min()) / (channel_np.max() - channel_np.min() + 1e-8)
        channel_np = (channel_np * 255).astype(np.uint8)

        save_path = os.path.join(output_dir, name+f'_c{c}.png')
        cv2.imwrite(save_path, channel_np)
        print(f"已保存: {save_path}")
if __name__ =="__main__":
    root_path_train = 'nyu_data/'
    root_path_train2 = 'CIDIS_200/train/'
    root_path_test = 'NYUV2/test/'
    root_path_test2 = 'CIDIS_200/test/'
    root_path_test_Middle = 'Middle/'
    root_path_test_Lu = 'Lu/'
    root_path_test_RGBDD = 'RGB-D-D/'
    patch_size = 128
    scale = 4
    dr_dataset_train = Dataset_CIDIS(dataset='CIDIS_200',path=root_path_train2, patch=patch_size, scale=8)
    loader_train = DataLoader(dr_dataset_train, batch_size=1, num_workers=0, shuffle=True)
    dr_dataset_test = Dataset_CIDIS(dataset='CIDIS_200',path=root_path_test2, patch=patch_size, scale=8, train=False)
    loader_test = DataLoader(dr_dataset_test, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_train = Dataset_NYU(path=root_path_train,patch=patch_size, scale=4)
    # loader_train = DataLoader(dr_dataset_train, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_test_NYU = Dataset_NYU(path=root_path_test, patch=patch_size, scale=4,train=False)
    # loader_test_NYU = DataLoader(dr_dataset_test_NYU, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_test_Middle = Dataset_Middle_LU(path=root_path_test_Middle, scale=4)
    # loader_test_Middle = DataLoader(dr_dataset_test_Middle, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_test_Lu = Dataset_Middle_LU(path=root_path_test_Lu, scale=4)
    # loader_test_Lu = DataLoader(dr_dataset_test_Lu, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_train_RGBDD = Dataset_RGBDD(path=root_path_test_RGBDD,patch=patch_size, scale=4,syn=True,train=True)
    # loader_train_RGBDD = DataLoader(dr_dataset_train_RGBDD, batch_size=1, num_workers=0, shuffle=True)
    # dr_dataset_test_RGBDD = Dataset_RGBDD(path=root_path_test_RGBDD, patch=patch_size, scale=4, syn=True,train=False)
    # loader_test_RGBDD = DataLoader(dr_dataset_test_RGBDD, batch_size=1, num_workers=0, shuffle=True)
    for packs in tqdm(loader_train):
        input1, input2, input1_gt = packs
        print(input1.shape)
        print(input2.shape)
        print(input1_gt.shape)


