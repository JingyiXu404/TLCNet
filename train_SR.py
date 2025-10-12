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
from dataset_mm import Dataset_Middle_LU,Dataset_RGBDD,NYU_v2_datset
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
import freeze

warnings.filterwarnings(action='ignore')
matplotlib.use('agg')

from model import PxAy as TLCNet

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
def data_process_npy(data_in,times=255):
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
    parser.add_argument("--epoch", dest="EPOCH", default=1500, type=int, help="epochs")
    parser.add_argument("--phase", dest="phase", default='train', type=str, help="train or test")
    parser.add_argument("--patch-size", dest="patch_size", default=64, type=int, help="size of train and test dataset")
    parser.add_argument("--layer", dest="layer", default=2, type=int, help="number of layers")
    parser.add_argument("--layer_Pz", dest="layer_Pz", default=2, type=int, help="number of layers")
    parser.add_argument("--hsize", dest="hsize", default=3, type=int, help="size of h kerner")
    parser.add_argument("--modelname", dest="model_name", default="PxAy", type=str, help="PxAy or else")
    parser.add_argument('--gpu_id', type=str, default=None, help='train use gpu')  #
    parser.add_argument('--load_pre', type=str, default="", help='train from checkpoints')
    parser.add_argument('--PSNR', type=str, default='10', help='Current best PSNR')
    parser.add_argument('--testset', dest="testset", type=str, default='NYU', help='Overlapping of different tiles')

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
def get_freeze_l2(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.freeze_by_names(net, ('head'))
    freeze.freeze_by_names(net, ('hypa_list'))
    freeze.freeze_by_names(net, ('update'))
def get_freeze_H(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.freeze_by_names(net, ('head'))
    freeze.freeze_by_names(net, ('hypa_list'))
    freeze.freeze_by_names(net, ('update'))
    freeze.unfreeze_by_names(net.update, ('0.generate_h'))
    freeze.unfreeze_by_names(net.update, ('1.generate_h'))
    freeze.unfreeze_by_names(net.update, ('2.generate_h'))


def get_unfreeze_l2(net):
    if isinstance(net, nn.DataParallel):
        net = net.module
    freeze.unfreeze_by_names(net, ('head'))
    freeze.unfreeze_by_names(net, ('hypa_list'))
    freeze.unfreeze_by_names(net, ('update'))

def process_networks(params):
    device = torch.device('cuda' if params.gpu_id is not None else 'cpu')
    if params.model_name == "TLCNet":
        model = TLCNet(num_of_layers=params.layer,h_size=params.hsize)
    else:
        pass
    if params.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
        print('USE GPU- ', params.gpu_id)
        model = nn.DataParallel(model).to(device)
        if (params.load_pre is not None):
            if os.path.exists(params.load_pre):
                checkpoint = torch.load(params.load_pre)
                new_checkpoint = {}
                for k in checkpoint.keys():
                    if ('hypa_list' in k):
                        # pass
                        new_checkpoint[k] = checkpoint[k]
                    if ('generate_h' in k):
                        # pass
                        new_checkpoint[k] = checkpoint[k]
                    if ('up_b' in k):
                        # pass
                        new_checkpoint[k] = checkpoint[k]
                    else:
                        new_checkpoint[k] = checkpoint[k]
                model.load_state_dict(new_checkpoint, strict=False)
                print('load model from ', params.load_pre)

    EPOCH = params.EPOCH
    phase = params.phase
    scale = params.scale
    testname = params.testset

    if not os.path.exists("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/"):
        os.makedirs("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/")
    if not os.path.exists("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/saved_models/"):
        os.makedirs("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/saved_models/")
    txtfile = "./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/results.txt"
    if not os.path.exists("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/loss_curve/"):
        os.makedirs("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/loss_curve/")
    if not os.path.exists("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/result_images/"):
        os.makedirs("./debug/SR" +str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+"/result_images/")

    print("loading dataset-------------------------------------")
    root_path_train_NYU = '/mnt/ssd1/XJY/GDSR/nyu_data/'
    root_path_test_NYU = '/mnt/ssd1/XJY/GDSR/nyu_data/'
    root_path_test_Middle = '/mnt/ssd1/XJY/GDSR/Middle/'
    root_path_test_Lu = '/mnt/ssd1/XJY/GDSR/Lu/'
    root_path_RGBDD = '/mnt/ssd1/XJY/GDSR/RGB-D-D/test2'
    dr_dataset_train = NYU_v2_datset(root_dir=root_path_train_NYU, patch=params.patch_size, scale=scale)
    train_loader = DataLoader(dr_dataset_train, batch_size=params.batch_size_train, num_workers=4, shuffle=True)
    dr_dataset_test_NYU = NYU_v2_datset(root_dir=root_path_test_NYU, patch=params.patch_size, scale=scale,train=False)
    test_loader_NYU = DataLoader(dr_dataset_test_NYU, batch_size=params.batch_size_test, num_workers=4, shuffle=False)
    dr_dataset_test_Middle = Dataset_Middle_LU(path=root_path_test_Middle, scale=scale)
    loader_test_Middle = DataLoader(dr_dataset_test_Middle, batch_size=params.batch_size_test, num_workers=0, shuffle=False)
    dr_dataset_test_Lu = Dataset_Middle_LU(path=root_path_test_Lu, scale=scale)
    loader_test_Lu = DataLoader(dr_dataset_test_Lu, batch_size=params.batch_size_test, num_workers=0, shuffle=False)
    dr_dataset_test_RGBDD = Dataset_Middle_LU(path=root_path_RGBDD, scale=scale)
    loader_test_RGBDD = DataLoader(dr_dataset_test_RGBDD, batch_size=params.batch_size_test, num_workers=0, shuffle=False)
    test_minmax = np.load('%s/test_minmax.npy' % root_path_train_NYU)
    data_loaders = {'train': train_loader, 'test_NYU': test_loader_NYU, 'test_Middle': loader_test_Middle, 'test_Lu': loader_test_Lu, 'test_RGBDD': loader_test_RGBDD}#}

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)#, weight_decay=1e-4
    MAE_Loss = nn.L1Loss().to(device)
    best_RMSE = [float(a) for a in params.PSNR.split(',')]
    best_epoch = [1, 1, 1,1]
    current_RMSE = [0,0,0,0]


    if phase == 'train':
        with open(txtfile, "a+") as file:
            plot_loss = []
            plot_batchPSNR = []
            plot_PSNR = []
            for epoch in range(1,EPOCH+1):
                train_bar = tqdm(data_loaders[phase])
                if epoch % 250 == 0:
                    new_lr = params.lr / (2*(epoch//250))
                    for para_group in optimizer.param_groups:
                        para_group['lr'] = new_lr
                    print("Learning weight decays to %f"%(new_lr))
                epoch_loss = []
                batch_psnr = []
                batch_rmse = []
                batch_ind = 0
                for target,guide, gt in train_bar:
                    # if epoch<200:
                    #     get_freeze_H(model)
                    # else:
                    #     get_unfreeze_l2(model)
                    batch_ind += 1
                    model = model.train()
                    # print(model.module.update[0].dropout.training)
                    target = target.to(device)
                    guide = guide.to(device)
                    gt =  gt.to(device)

                    optimizer.zero_grad()

                    output, output_else = model(target,guide)

                    total = gt.size(0)
                    loss1 = cal_multi_loss(MAE_Loss,output,gt)
                    loss_Py = cal_multi_loss(MAE_Loss,[get_PA(i)[0] for i in output],get_PA(guide)[0])
                    loss_Ax = cal_multi_loss(MAE_Loss,[get_PA(i)[1] for i in output],get_PA(target)[1])
                    loss2 = 0.5*loss_Py+0.5*loss_Ax
                    loss = loss1 #+ 0.001 * loss2
                    loss.backward()
                    optimizer.step()

                    output = data_process_npy(output[-1])
                    gt = data_process_npy(gt)
                    # print(output.shape,gt.shape)
                    epoch_loss.append(loss.item())
                    batch_psnr.append(round(psnr.psnr(output,gt),2))
                    batch_rmse.append(round(psnr.RMSE(output,gt),2))
                    train_bar.set_description(desc= ' [%d/%d] ce_loss: %.4f=%.4f+0.001 %.4f  | batch_psnr: %.4f | batch_rmse: %.4f ' % (epoch, EPOCH,loss.item(),loss1.item(),loss2.item(),round(psnr.psnr(output,gt),2),round(psnr.RMSE(output,gt),2)))
                save = './debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/saved_models/latest.pth'
                torch.save(model.state_dict(), save)


                fig1 = plt.figure()
                plot_loss.append(np.mean(epoch_loss))
                plt.plot(plot_loss)
                plt.savefig('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/loss_curve/MAE_loss.png')
                fig2 = plt.figure()
                plot_batchPSNR.append(np.mean(batch_psnr))
                plt.plot(plot_batchPSNR)
                plt.savefig('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/loss_curve/PSNR_batchwise.png')

                # eval
                if (epoch < 50 and epoch%1==0) or epoch>=50:
                    test_datasets = ['NYU','Middle', 'Lu', 'RGBDD']#
                    count = 0
                    for dataset in test_datasets:
                        rmse_list = []
                        psnr_list = []
                        ssim_list = []
                        with torch.no_grad():
                            val_bar = tqdm(data_loaders['test_'+dataset])
                            y = []
                            pred = []
                            batch_ind = 0
                            for target,guide, gt in val_bar:
                                if dataset == 'NYU':
                                    minmax = test_minmax[:, batch_ind]
                                    minmax = torch.from_numpy(minmax).to(device)
                                batch_ind += 1
                                model = model.eval()
                                # print(model.module.update[0].dropout.training)

                                noise = data_process_npy(target)
                                target = target.to(device)
                                guide = guide.to(device)
                                gt = gt.to(device)
                                h, w = gt.size()[-2:]

                                output, output_else = model(target,guide)
                                out = output[-1][..., :h, :w]
                                if dataset == 'NYU':
                                    rmse = calc_rmse(gt[0, 0], out[0, 0], minmax)
                                    rmse_list.append(rmse.cpu().numpy())
                                else:
                                    rmse = midd_calc_rmse(gt[0, 0], out[0, 0])
                                    rmse_list.append(rmse.cpu().numpy())

                                out = data_process_npy(out)
                                gt = data_process_npy(gt)

                                psnr_list.append(psnr.psnr(gt, out))
                                ssim_list.append(psnr.SSIM(gt, out))

                            PSNR = np.mean(psnr_list)
                            RMSE = np.mean(rmse_list)
                            SSIM = np.mean(ssim_list)
                            current_RMSE[count] = RMSE
                            if RMSE < best_RMSE[count]:
                                best_RMSE[count] = RMSE
                                best_epoch[count] = epoch
                                print("saving best model of "+dataset+".....")
                                save = './debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/saved_models/best_'+dataset+'.pth'
                                torch.save(model.state_dict(), save)

                            if epoch % 100 == 0:
                                print("saving model.....")
                                save= './debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/saved_models/epoch_'+str(epoch)+'.pth'
                                torch.save(model.state_dict(), save)
                        count = count+1
                        fig3 = plt.figure()
                        plot_PSNR.append(PSNR)
                        plt.plot(plot_PSNR)
                        plt.savefig('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name+'_l'+str(params.layer)+'/loss_curve/PSNR.png')
                    print('Epoch: {} RMSE: {} ####  bestRMSE: {} bestEpoch: {}'.format(epoch, current_RMSE, best_RMSE, best_epoch))
                    file.write('#TEST#:Epoch:{} RMSE: {} #### bestEpoch:{} bestRMSE:{}'.format(epoch, current_RMSE, best_RMSE, best_epoch))
                    file.write('\n')
                    file.flush()

    elif phase == 'test':
        dict_name = ['m','n','b','v','u','w','H']
        os.makedirs('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_best/',exist_ok=True)
        # os.makedirs('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_target/',exist_ok=True)
        # os.makedirs('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_gt/',exist_ok=True)
        # os.makedirs('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_guide/',exist_ok=True)
        # os.makedirs('./debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_a/',exist_ok=True)
        with torch.no_grad():
            val_bar = tqdm(data_loaders['test_'+testname])
            batch_ind = 0
            rmse_list = []
            psnr_list = []
            ssim_list = []
            for data in val_bar:
                target, guide, gt = data[0].to(device),data[1].to(device),data[2].to(device)
                if testname=='NYU':
                    minmax = test_minmax[:, batch_ind]
                    minmax = torch.from_numpy(minmax).to(device)
                batch_ind += 1
                model = model.eval()
                h, w = gt.size()[-2:]
                output, output_else = model(target,guide)
                for i in range(len(output)):
                    zi = data_process_npy(output[i][...,:h,:w])
                    os.makedirs('./debug/SR' + str(scale) + "_p" + str(params.patch_size) + "_" + params.model_name + '_l' + str(params.layer)  + '/result_images/' + testname + '_z'+str(i+1)+'/',exist_ok=True)
                    path_output_zi = './debug/SR' + str(scale) + "_p" + str(params.patch_size) + "_" + params.model_name + '_l' + str(params.layer)  + '/result_images/' + testname + '_z'+str(i+1)+'/' + f"{batch_ind:03d}" + '.png'
                    cv2.imwrite(path_output_zi, zi)
                # for i in range(len(output_else)):
                #     zi = output_else[i]
                #     for j in range(len(zi)):
                #         zii = data_process_npy(zi[j])
                #         os.makedirs('./debug/SR' + str(scale) + "_p" + str(params.patch_size) + "_" + params.model_name + '_l' + str(params.layer)  + '/result_images/' + testname + '_'+dict_name[j]+str(i+1)+'/',exist_ok=True)
                #         path_output_zi = './debug/SR' + str(scale) + "_p" + str(params.patch_size) + "_" + params.model_name + '_l' + str(params.layer)  + '/result_images/' + testname + '_'+dict_name[j]+str(i+1)+'/' + f"{batch_ind:03d}" + '.png'
                #         cv2.imwrite(path_output_zi, zii)

                out = output[-1][..., :h, :w]
                if testname =='NYU':
                    rmse = calc_rmse(gt[0,0],out[0,0],minmax)
                    rmse_list.append(rmse.cpu().numpy())
                else:
                    rmse = midd_calc_rmse(gt[0, 0], out[0, 0])
                    rmse_list.append(rmse.cpu().numpy())

                out = data_process_npy(out)
                gt = data_process_npy(gt)
                target = data_process_npy(target)
                guide = data_process_npy(guide)

                psnr_list.append(psnr.psnr(gt,out))
                ssim_list.append(psnr.SSIM(gt,out))

                path_output = './debug/SR'+str(scale)+"_p"+str(params.patch_size)+"_"+params.model_name +'_l'+str(params.layer)+ '/result_images/'+testname+'_best/' + f"{batch_ind:03d}" + '.png'
                cv2.imwrite(path_output, out)

            PSNR = np.mean(psnr_list)
            RMSE = np.mean(rmse_list)
            SSIM = np.mean(ssim_list)

            print("Test: PSNR=%.2fdB RMSE=%.4f SSIM=%.4f" % (PSNR,RMSE,SSIM))


if __name__ == '__main__':
    params = get_overall_run_params()
    process_networks(params)