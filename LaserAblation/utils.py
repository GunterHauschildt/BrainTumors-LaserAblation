import torch
import time
import os
import shutil
from scipy import ndimage as ndi
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pt'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pt')


def soft_laser_beam_3d_batched(r0, a0, s0, r1, a1, s1, R, A, S,
                               sigma=1.5, tau=0.05, device='cuda'):

    B = r0.shape[0]

    Rv, Av, Sv = torch.meshgrid(
        torch.arange(R, device=device),
        torch.arange(A, device=device),
        torch.arange(S, device=device),
        indexing='ij'
    )
    Rv = Rv.float()[None].to(device)
    Av = Av.float()[None].to(device)
    Sv = Sv.float()[None].to(device)  # (1,S,A,R)

    r0 = r0[:, None, None, None].to(device)
    a0 = a0[:, None, None, None].to(device)
    r1 = r1[:, None, None, None].to(device)
    a1 = a1[:, None, None, None].to(device)

    s0 = s0[None, None, None, None].expand(B, 1, 1, 1)
    s1 = s1[None, None, None, None].expand(B, 1, 1, 1)

    dR = r1 - r0
    dA = a1 - a0
    dS = s1 - s0
    denom = dR*dR + dA*dA + dS*dS + 1e-8

    t = ((Rv - r0) * dR + (Av - a0) * dA + (Sv - s0) * dS) / denom

    Rp = r0 + t * dR
    Ap = a0 + t * dA
    Sp = s0 + t * dS

    dist2 = (Rv-Rp)**2 + (Av-Ap)**2 + (Sv-Sp)**2
    beam = torch.exp(-dist2 / (2*sigma**2))

    seg = torch.sigmoid(t/tau) * torch.sigmoid((1-t)/tau)

    return beam * seg   # (B, R, A, S)


def keep_largest_components_3d(labels, target_label):
    """
    labels: 3D numpy array of ints
    target_label: the label to filter
    """
    mask = (labels == target_label)

    struct = ndi.generate_binary_structure(3, 1)
    cc, n = ndi.label(mask, structure=struct)

    max = 0
    p_max = 0
    for i in range(1, n):
        mask = np.where(cc == i, 1, 0)
        if (size := np.count_nonzero(mask)) > max:
            max = size
            p_max = i

    return np.where(cc == p_max, 1, 0)
