import torch
import torch.nn.functional as F
from torch.nn.functional import pad
import torch.nn as nn


def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]

    a = a * (minmax[0] - minmax[1]) + minmax[1]
    b = b * (minmax[0] - minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100

    return torch.sqrt(torch.mean(torch.pow(a - b, 2)))


def rgbdd_calc_rmse(gt, out, minmax):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    # gt = gt*(minmax[0]-minmax[1]) + minmax[1]
    out = out * (minmax[0] - minmax[1]) + minmax[1]
    gt = gt / 10.0
    out = out / 10.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))


def midd_calc_rmse(gt, out):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))
def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def csum(x, y):
    # complex + real
    real = x[..., 0]
    real = real + y[..., 0].expand_as(real)
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)


def cabs2(x):
    return x[..., 0]**2 + x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1).cuda()


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    kernel_size = (psf.size(-2), psf.size(-1))
    psf = F.pad(psf,[0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])
    # print('0',psf.shape)
    psf = roll(psf, kernel_size)
    # print('1',psf.shape)
    psf = torch.fft.rfft2(psf, dim=(-1),norm = "ortho")
    psf = torch.stack((psf.real, psf.imag), -1)
    # print('2',psf.shape)
    return psf


def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,int(axis_size / 2) * (-1 if not reverse else 1),dims=axis)
    return psf


def conv2d(input, weight, padding=0, sample_wise=False):
    """
        sample_wise=False, normal conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (C_out, C_in, H_k, W_k)
        sample_wise=True, sample-wise conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (N, C_out, C_in, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    if sample_wise:
        # input - (N, C_in, H_in, W_in) -> (1, N * C_in, H_in, W_in)
        input_sw = input.view(1,input.size(0) * input.size(1), input.size(2),input.size(3))

        # weight - (N, C_out, C_in, H_k, W_k) -> (N * C_out, C_in, H_k, W_k)
        weight_sw = weight.view(weight.size(0) * weight.size(1), weight.size(2), weight.size(3),weight.size(4))

        # group-wise convolution, group_size==batch_size
        out = F.conv2d(pad(input_sw, padding, mode='circular'),weight_sw,groups=input.size(0))
        out = out.view(input.size(0), weight.size(1), out.size(2), out.size(3))
    else:
        out = F.conv2d(pad(input, padding, mode='circular'), weight)
    return out


def conv3d(input, weight, padding=0, sample_wise=False):
    """
        sample_wise=False, normal conv3d:
            input - (N, C_in, D_in, H_in, W_in)
            weight - (C_out, C_in, D_k, H_k, W_k)
        sample_wise=True, sample-wise conv3d:
            input - (N, C_in, D_in, H_in, W_in)
            weight - (N, C_out, C_in, D_k, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4 + [0, 0]
    if sample_wise:
        # input - (N, C_in, D_in, H_in, W_in) -> (1, N * C_in, D_in, H_in, W_in)
        input_sw = input.view(1,input.size(0) * input.size(1), input.size(2),input.size(3), input.size(4))

        # weight - (N, C_out, C_in, D_k, H_k, W_k) -> (N * C_out, C_in, D_k, H_k, W_k)
        weight_sw = weight.view(weight.size(0) * weight.size(1), weight.size(2), weight.size(3),weight.size(4), weight.size(5))

        # group-wise convolution, group_size==batch_size
        out = F.conv3d(pad(input_sw, padding, mode='circular'),weight_sw,groups=input.size(0))
        out = out.view(input.size(0), weight.size(1), out.size(2), out.size(3),out.size(4))
    else:
        out = F.conv3d(pad(input, padding, mode='circular'),weight,padding=padding)
    return out


def unfold5d(x, kernel_size):
    """perform 2D unfold on (the last 2 dimensions of) 5D Tensor"""
    x_reshape = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
    x_unfold = F.unfold(x_reshape, kernel_size)
    x_unfold = x_unfold.view(x.size(0), x.size(1), x_unfold.size(1),x_unfold.size(2))
    return x_unfold
def reshape_params(lambda1,Z):
    lambda1 = lambda1.unsqueeze(1).unsqueeze(1) / Z.size(2)
    lambda1 = lambda1.view(lambda1.size(0), lambda1.size(1), lambda1.size(2), lambda1.size(3),lambda1.size(4)// 2, 2)
    return lambda1[:,:,:,:Z.size(3),:,:]
def reshape_params3(lambda1,Z):
    lambda1 = lambda1.unsqueeze(1)/ Z.size(2)
    lambda1 = lambda1.view(lambda1.size(0), lambda1.size(1), lambda1.size(2), lambda1.size(3),lambda1.size(4)// 2, 2)
    return lambda1[:,:,:,:Z.size(3),:,:]
def reshape_params2(lambda1,z):
    lambda1 = lambda1.reshape(lambda1.size(0), lambda1.size(3), lambda1.size(1), lambda1.size(2))
    return lambda1[:,:,:z.size(2),:z.size(3)]
def reshape_params4(lambda1,z):
    lambda1 = lambda1.reshape(lambda1.size(0), lambda1.size(1), lambda1.size(2), lambda1.size(3))
    return lambda1[:,:,:z.size(2),:z.size(3)]

def amp_pha(x):
    xp = torch.fft.fft2(x)# xp = fftshift(xp)
    real_part = xp.real
    imag_part = xp.imag
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    phase = torch.angle(xp)
    return magnitude, phase #xm, xp

def FFT_PxAy_IFFT(x,y):
    Ax, Px = amp_pha(x)
    Ay, Py = amp_pha(y)
    phase = torch.cos(Px) + 1j * torch.sin(Px)
    amplitude = Ay
    out = torch.fft.ifft2(phase*amplitude).real
    return out.squeeze(-1)
def FFT_PyAx_IFFT(x,y):
    Ax, Px = amp_pha(x)
    Ay, Py = amp_pha(y)
    phase = torch.cos(Py) + 1j * torch.sin(Py)
    amplitude = Ax
    out = torch.fft.ifft2(phase*amplitude).real
    return out.squeeze(-1)
def IFFT_xp(xp):
    phase_spectrum = torch.cos(xp) + 1j * torch.sin(xp)# phase_spectrum_shift = ifftshift(phase_spectrum)
    out_xp = torch.fft.ifft2(phase_spectrum).real
    return out_xp.squeeze(-1)
def IFFT_zm(zm):
    epsilon = 1e-10
    Lf = zm
    # resx_shift = ifftshift(Lf)
    out_zm = torch.fft.ifft2(Lf).real
    return out_zm.squeeze(-1)
def FFT_PzAz_IFFT(Pz,zm):
    zp = torch.fft.fft2(Pz)
    out = torch.fft.ifft2(zp*zm).real
    return out.squeeze(-1)
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)