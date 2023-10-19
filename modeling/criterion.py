import torch
import torch.nn.functional as F

import numpy as np


def loss_single_double(pred: dict, ddg_dense1, ddg_dense2, batch, args, train) -> dict:
    # sample fewer destabilizing mutations
    stbl_ratio = args.double_subsample_destabilizing_ratio
    if stbl_ratio > 0 and train:
        for b in range(len(ddg_dense2)):
            ddg_dense_b = ddg_dense2[b]
            destbl_inds = ((ddg_dense_b != 999) & (ddg_dense_b > 0)).nonzero()
            n_destbl = len(destbl_inds)
            n_stbl = (ddg_dense_b < 0).sum().item() + 1
            if n_destbl < stbl_ratio * n_stbl:
                continue
            mask_inds = np.random.choice(n_destbl, n_destbl - int(stbl_ratio * n_stbl), replace=False)
            ddg_dense2[b][destbl_inds[mask_inds].split(1,dim=1)] = 999

    # mask unknown values
    unknown_mask1 = ddg_dense1 == 999
    unknown_mask2 = ddg_dense2 == 999

    losses = {}

    if unknown_mask1.all():
        losses['loss1'] = 0. * pred['mut1_ddg'].sum()
    else:
        losses['loss1'] = F.huber_loss(pred['mut1_ddg'][~unknown_mask1], ddg_dense1[~unknown_mask1]) * args.lambda_single

    if args.multi_dec == 'epistasis':
        # unknown if any of (ddg_ij, ddg_i, ddg_j) are unknown
        unknown_mask2 |= (unknown_mask1[:,None,None,:,:] | unknown_mask1[:,:,:,None,None])

    if ~unknown_mask2.all():
        pos_mask = (ddg_dense2 <= 0)[~unknown_mask2]
        weight2 = torch.cat([n_b.new_ones(n_b) / n_b for n_b in (~unknown_mask2).flatten(1,-1).sum(1)])
        weight2 *= 1 + pos_mask * (args.lambda_pos - 1)
        losses2 = F.huber_loss(pred['mut2_ddg'][~unknown_mask2], ddg_dense2[~unknown_mask2], reduction='none')
        loss2 = (losses2 * weight2).sum() / weight2.sum()
        losses['loss2'] = loss2 * args.lambda_double
    elif 'mut2_ddg' not in pred:
        losses['loss2'] = losses['loss1'] * 0.
    else:
        losses['loss2'] = 0. * pred['mut2_ddg'].sum()
    return losses
