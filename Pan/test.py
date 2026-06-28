import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import platform
from time import *
from tqdm import tqdm, trange
import torch.cuda
from dataset_mm import Dataset_Middle_LU,Dataset_RGBDD,NYU_v2_datset,Dataset_CIDIS,Dataset_pansharpen
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import argparse
import warnings
import psnr
import math
import cv2
from utils import *
from metrics.metrics import ref_evaluate,no_ref_evaluate,window_partitionx,window_reversex
from einops import rearrange

warnings.filterwarnings(action='ignore')
matplotlib.use('agg')


from TLCNet import TLCNet as TLCNet

def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x
def pad(img):
    h, w = img.shape[2], img.shape[3]
    h_psz = padding_size(h, 4)
    w_psz = padding_size(w, 4)
    padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
    img = padding(img)
    return img
def data_process_npy(data_in,times=1):
    data_out=data_in.detach().float().cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out = data_out.squeeze()
    return data_out*times
def get_PA(data):
    magnitude, phase = amp_pha(data)  # 做傅里叶变换得到幅度谱和相位谱
    P_data = IFFT_xp(phase)  # 相位谱逆变换
    A_data = IFFT_zm(magnitude)  # 幅度谱逆变换
    return P_data, A_data
def get_overall_run_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size-train", dest="batch_size_train", default=16, type=int)
    parser.add_argument("--batch-size-test", dest="batch_size_test", default=1, type=int)
    parser.add_argument("--scale", dest="scale", default=4, type=int)
    parser.add_argument("--lr", dest="lr", default=1 * 1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch", dest="EPOCH", default=1000, type=int, help="epochs")
    parser.add_argument("--phase", dest="phase", default='train', type=str, help="train or test")
    parser.add_argument("--layer", dest="layer", default=3, type=int, help="number of layers")
    parser.add_argument("--modelname", dest="model_name", default="TLCNet", type=str, help="PxAy or else")
    parser.add_argument('--gpu_id', type=str, default=None, help='train use gpu')  #
    parser.add_argument('--load_pre', type=str, default="cpts/", help='train from checkpoints')
    parser.add_argument('--PSNR', type=str, default='10', help='Current best PSNR')
    parser.add_argument('--testset', dest="testset", type=str, default='gf2', help='Overlapping of different tiles')

    params = parser.parse_args()
    return params

def cal_multi_loss(lossfn, preds, gt):
    losses = None
    for i, pred in enumerate(preds):
        loss = lossfn(pred, gt)
        if i != len(preds) - 1:
            loss *= (1 / (len(preds) - 1))
        if i == 0:
            losses = loss
        else:
            losses += loss
    return losses

def process_networks(params):
    device = torch.device('cuda' if params.gpu_id is not None else 'cpu')
    EPOCH = params.EPOCH
    phase = params.phase
    scale = int(params.scale)
    testname = params.testset
    if 'wv' in testname:
        in_channel = 8
    else:
        in_channel = 4
    if params.model_name == "TLCNet":
        model = TLCNet(num_of_layers=params.layer,in_channels=in_channel,out_channels=in_channel)
    else:
        pass
    if params.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
        print('USE GPU- ', params.gpu_id)
        model = nn.DataParallel(model).to(device)
        model_path = params.load_pre+'best_'+testname+'.pth'
        print(model_path)
        if (model_path is not None):
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                new_checkpoint = {}
                for k in checkpoint.keys():
                    new_checkpoint[k] = checkpoint[k]
                model.load_state_dict(new_checkpoint, strict=True)
                print('load model from ', model_path)

    print("loading dataset-------------------------------------")

    if 'wv' in testname:
        root_path_train = '/mnt/ssd1/XJY/GDSR/Pansharpening/PanCollection/training_data/train_wv3.h5'
        root_path_test_wv2 = '/mnt/ssd1/XJY/GDSR/Pansharpening/PanCollection/test_data/wv2/test_wv2_multiExm1.h5'
        root_path_test_wv3 = '/mnt/ssd1/XJY/GDSR/Pansharpening/PanCollection/test_data/wv3/test_wv3_multiExm1.h5'
        dr_dataset_train = Dataset_pansharpen(path=root_path_train, img_scale=scale, train=True)
        train_loader = DataLoader(dr_dataset_train, batch_size=params.batch_size_train, num_workers=4, shuffle=True)
        dr_dataset_test_wv2 = Dataset_pansharpen(path=root_path_test_wv2, img_scale=scale, train=False)
        test_loader_Pan_wv2 = DataLoader(dr_dataset_test_wv2, batch_size=params.batch_size_test, num_workers=4, shuffle=False)
        dr_dataset_test_wv3 = Dataset_pansharpen(path=root_path_test_wv3, img_scale=scale, train=False)
        test_loader_Pan_wv3 = DataLoader(dr_dataset_test_wv3, batch_size=params.batch_size_test, num_workers=4, shuffle=False)
        data_loaders = {'train': train_loader, 'test_wv2' : test_loader_Pan_wv2, 'test_wv3' : test_loader_Pan_wv3}  # }
    else:
        root_path_train = '/mnt/ssd1/XJY/GDSR/Pansharpening/PanCollection/training_data/train_'+str(testname)+'.h5'
        root_path_test = '/mnt/ssd1/XJY/GDSR/Pansharpening/PanCollection/test_data/'+str(testname)+'/test_'+str(testname)+'_multiExm1.h5'
        print(root_path_train)
        dr_dataset_train = Dataset_pansharpen(path=root_path_train, img_scale=scale, train=True)
        train_loader = DataLoader(dr_dataset_train, batch_size=params.batch_size_train, num_workers=4, shuffle=True)
        dr_dataset_test = Dataset_pansharpen(path=root_path_test, img_scale=scale, train=False)
        test_loader_Pan = DataLoader(dr_dataset_test, batch_size=params.batch_size_test, num_workers=4, shuffle=False)
        data_loaders = {'train': train_loader, 'test_'+testname: test_loader_Pan}#}

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)#, weight_decay=1e-4
    MAE_Loss = nn.L1Loss().to(device)
    best_PSNR = [float(a) for a in params.PSNR.split(',')]
    best_epoch = [1]
    current_PSNR = [0]


    if phase == 'train':
        pass
    elif phase == 'test':
        os.makedirs('Results/' + params.model_name + '_l' + str(params.layer) + '_' + testname + '/', exist_ok=True)

        with torch.no_grad():

            val_bar = tqdm(data_loaders['test_'+testname])
            batch_ind = 0
            tmp_results = {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'SCC': [], 'Q': []}

            for input_data in val_bar:
                target, guide, gt = input_data["lms"].to(device), input_data["pan"].repeat(1, in_channel, 1, 1).to(device), input_data["gt"].to(device)
                batch_ind += 1
                model = model.eval()
                h, w = gt.size()[-2:]
                output, output_else = model(target,guide)
                out = output[-1][..., :h, :w]
                # print(out.max(),out.min())
                out = torch.clip(out, 0., 1.)
                path_output = 'Results/' + params.model_name + '_l' + str(params.layer) + '_' + testname + '/' + f"{batch_ind:03d}" + '.png'
                cv2.imwrite(path_output, (to_rgb(out.squeeze(0))* 255).astype(np.uint8))
                out = data_process_npy(out)
                gt = data_process_npy(gt)

                # print(out.max(),out.min(),gt.max(),gt.min(),target.max(),target.min(),guide.max(),guide.min())
                results = ref_evaluate(out, gt)
                tmp_results['SAM'].append(results["SAM"])
                tmp_results['SCC'].append(results["SCC"])
                tmp_results['Q'].append(results["Q"])
                tmp_results['ERGAS'].append(results["ERGAS"])
                tmp_results['PSNR'].append(results["PSNR"])
                tmp_results['SSIM'].append(results["SSIM"])



            PSNR = np.mean(tmp_results['PSNR'])
            SSIM = np.mean(tmp_results['SSIM'])
            SAM = np.mean(tmp_results['SAM'])
            SCC = np.mean(tmp_results['SCC'])
            Q = np.mean(tmp_results['Q'])
            ERGAS = np.mean(tmp_results['ERGAS'])

            print("Test: PSNR=%.2fdB SSIM=%.4f SAM=%.4f SCC=%.4f Q=%.4f ERGAS=%.4f" % (PSNR,SSIM,SAM,SCC,Q,ERGAS))

def save_diff(img, gt, folder, name):
        os.makedirs(folder, exist_ok=True)
        diff = np.abs(img - gt)  # 计算绝对差
        diff = np.mean(diff, axis=2)
        plt.switch_backend('Agg')
        fig, ax = plt.subplots()
        ax.imshow(diff, cmap='viridis', vmin=0, vmax=0.22)
        ax.axis('off')
        plt.savefig(os.path.join(folder, name), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图像窗口
def to_rgb(x, tol_low=0.01, tol_high=0.99):
        c = x.shape[0]
        if c == 4:
            x = x[[2, 1, 0], :, :]
        elif c == 8:
            x = x[[4, 2, 1], :, :]
        else:
            raise ValueError(f"Unsupported channel number: {c}")
        c, h, w = x.shape
        x = rearrange(x, 'c h w -> c (h w)')
        sorted_x, _ = torch.sort(x, dim=1)
        t_low = sorted_x[:, int(h * w * tol_low)].unsqueeze(1)
        t_high = sorted_x[:, int(h * w * tol_high)].unsqueeze(1)
        x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
        x = rearrange(x, 'c (h w) -> h w c',c=c, h=h, w=w)
        return x.cpu().numpy()
def save_tensor_channels(data, output_dir,name):
    os.makedirs(output_dir, exist_ok=True)
    num_channels = data.shape[-1]
    for c in range(num_channels):
        channel_dir = os.path.join(output_dir, f'c{c}')
        os.makedirs(channel_dir, exist_ok=True)
        channel = data[:, :, c]
        # out = (channel * 255).astype(np.uint8)
        save_path = os.path.join(channel_dir, name)
        cv2.imwrite(save_path, channel)
if __name__ == '__main__':
    params = get_overall_run_params()
    process_networks(params)