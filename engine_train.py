import math
import sys
import numpy as np

import torch
import torch.nn as nn

from data import get_dense_mut_infos, get_dense_double_mut_infos
from modeling.utils import mem_inputs_to_device
from modeling.criterion import loss_single_double
from metrics import eval_ddg
import misc
import wandb


def train_one_epoch(model: torch.nn.Module,
                    dl, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None):
    ## prepare training
    model.train(True)
    optimizer.zero_grad()

    ## prepare logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        ## move inputs/outputs to cuda
        x = mem_inputs_to_device(batch, device, args)
        ddg_dense1 = batch['ddg_dense'] = batch['ddg_dense'].to(device, non_blocking=True)
        ddg_dense2 = batch['ddg_dense2'] = batch['ddg_dense2'].to(device, non_blocking=True)

        ## forward
        pred = model(x, batch)
        losses = loss_single_double(pred, ddg_dense1, ddg_dense2, batch, args, True)
        loss = sum(losses.values())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        ## backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()

        ## logging
        lr = optimizer.param_groups[0]["lr"]
        losses_detach = {f'train_{k}': v.cpu().item() for k, v in losses.items()}
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss.item())
        metric_logger.update(**losses_detach)
        if not args.disable_wandb and misc.is_main_process():
            wandb.log({
                'train_loss': loss.item(),
                'lr': lr,
                **losses_detach,
            })

    ## gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, dl, device, args):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_preds = {}
    for batch in metric_logger.log_every(dl, 10, header):
        x = mem_inputs_to_device(batch, device, args)

        ## infer eval modes - single/double/arbitrary
        batch['known_mask1'] = known_mask1 = batch['ddg_dense'] != 999
        batch['known_mask2'] = known_mask2 = batch['ddg_dense2'] != 999
        eval_single = known_mask1.any()
        eval_double = known_mask2.any()
        eval_list = 'mut_info_list' in batch and sum([len(x) for x in batch['mut_info_list']]) > 0
        batch['only_eval_single'] = eval_single and not eval_double and not eval_list
        batch['eval_list'] = eval_list

        ## forward
        pred_dict = model(x, batch)
        pr1 = pred_dict['mut1_ddg']
        pr2 = pred_dict.get('mut2_ddg', None)

        ## format for eval
        for b in range(len(x)):
            pdb_id = batch['pdb_ids'][b]
            seq = batch['seqs'][b]
            muts, scores = [], []

            if eval_list:
                muts.append(np.array(batch['mut_info_list'][b]))
                scores.append(-pred_dict['pr_ddgs_list'][b].detach().cpu().numpy())

            if eval_double:
                mutations, valid_mask = get_dense_double_mut_infos(seq)  # slow
                pr_ddgs = pr2[b].flatten()
                keep_inds = known_mask2[b].flatten().cpu().numpy()
                muts.append(mutations[keep_inds & valid_mask])
                scores.append(-pr_ddgs[keep_inds & valid_mask].detach().cpu().numpy())

            if eval_single:
                mutations = np.array(get_dense_mut_infos(seq))
                pr_ddgs = pr1[b].flatten()
                keep_inds = known_mask1[b].flatten().cpu().numpy()
                muts.append(mutations[keep_inds])
                scores.append(-pr_ddgs[keep_inds].detach().cpu().numpy())

            all_preds[pdb_id] = {
                'mutations': np.concatenate(muts),
                'scores': np.concatenate(scores),
            }

    if args.dist_eval:
        print('Start gathering predictions')
        torch.cuda.empty_cache()
        all_preds = misc.gather_dict_keys_on_main(all_preds)
        print(f'Finished gathering predictions')
        if not misc.is_main_process():
            return {}

    ds_name = dl.dataset.name
    metrics, metrics_det, metrics_det_pdb, copypaste, merged_df = eval_ddg(dl.dataset.mutdf, all_preds)
    merged_df['ddg_pred'] = -merged_df['scores']
    print(f'Saving results to {args.output_dir}/results_{ds_name}.csv')
    print(f'Saving metrics to {args.output_dir}/metrics_{ds_name}.csv')
    merged_df.to_csv(f'{args.output_dir}/results_{ds_name}.csv', index=False)
    if metrics_det is not None:
        metrics_det_pdb.to_csv(f'{args.output_dir}/metrics_{ds_name}.csv', index=False)
        print(metrics_det)
    print(ds_name, copypaste)

    metric_logger.update(**metrics)
    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret['copypasta'] = copypaste
    ret = {f'{ds_name}_{k}': v for k, v in ret.items()}
    if not args.disable_wandb and misc.is_main_process():
        wandb.log(ret)
    return ret
