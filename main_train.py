import argparse
import datetime
import time
import wandb
import numpy as np
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim

from modeling.mutate_everything import MutateEverything
from engine_train import train_one_epoch, evaluate
from data import SeqDetDatatset, protein_collate_fn
import misc

def get_args_parser():
    parser = argparse.ArgumentParser('Train Sequence Detector')
    parser.add_argument('--seed', default=0, type=int)

    # model params
    parser.add_argument('--aa_expand', default='backbone', help='scratch|backbone')
    parser.add_argument('--single_dec', default='naive', help='naive')
    parser.add_argument('--multi_dec', default='epistasis', help='additive|epistasis')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--backbone', default='esm2_t33_650M_UR50D', help='af|esm2_t33_650M_UR50D')
    parser.add_argument('--finetune_backbone', type=str, default='models/finetuning_ptm_2.pt')
    parser.add_argument('--freeze_at', default=0, help='freeze backbone up to layer X')

    # af params
    parser.add_argument('--n_msa_seqs', type=int, default=128)
    parser.add_argument('--n_extra_msa_seqs', type=int, default=1024)
    parser.add_argument('--af_extract_feat', type=str, default='both',
        help='which features to use from AF: both|evo|struct')

    # data params
    parser.add_argument('--data_path', type=str, default='data/cdna_train.csv')
    parser.add_argument('--eval_data_paths', type=str,
        default='data/cdna2_test.csv,data/ptmul.csv,data/s669.csv',
        help='comma separated string of data paths to evaluate')
    parser.add_argument('--max_context_length', type=int, default=2000,
        help='max length of protein sequence')
    parser.add_argument('--num_workers', default=10, type=int)

    # train params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # loss params
    parser.add_argument('--lambda_single', type=float, default=0.1)
    parser.add_argument('--lambda_double', type=float, default=1.)
    parser.add_argument('--double_subsample_destabilizing_ratio', type=float, default=8)
    parser.add_argument('--lambda_pos', type=float, default=4)
    
    # eval params
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true')
    parser.add_argument('--test', action='store_true',
        help='when testing, please use data_path NOT eval_data_paths')

    # resume params
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', type=int, default=0)

    # logging params
    parser.add_argument('--output_dir', type=Path, default='logs/mutate_everything')
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--save_period', type=int, default=1000)
    parser.add_argument('--disable_wandb', action='store_true')

    # distributed training parameters
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)
    if not args.disable_wandb and misc.is_main_process():
        run_name = args.output_dir.name
        wandb.init(project='mutate_everything', name=run_name, config=args, dir=args.output_dir)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## prepare model
    model = MutateEverything(args)
    alphabet = model.backbone.get_alphabet()
    n_params = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Training {n_params_grad} of {n_params} parameters')

    ## prepare DDP
    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    ## prepare optimizer
    param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    misc.load_model(args, model_without_ddp, optimizer, None)

    ## prepare test data
    dls_test = []
    for eval_data_path in args.eval_data_paths.split(','):
        ds_test = SeqDetDatatset(eval_data_path, args, train=False)
        collate_fn = partial(protein_collate_fn, alphabet=alphabet, args=args)
        if args.distributed and args.dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_test = torch.utils.data.SequentialSampler(ds_test)
        dl_test = torch.utils.data.DataLoader(ds_test, sampler=sampler_test, batch_size=1, collate_fn=collate_fn)
        dls_test.append(dl_test)

    if args.eval:
        metrics = {}
        for dl_test in dls_test:
            metrics.update(evaluate(model, dl_test, device, args))
        if misc.is_main_process():
            metrics['copypasta'] = ',,'.join([metrics[f'{dl.dataset.name}_copypasta'] for dl in dls_test])
            print(metrics)
        if not args.disable_wandb and misc.is_main_process():
            wandb.log({'copypasta': metrics['copypasta']})
            wandb.finish()
        exit()

    ## prepare train data
    ds_train = SeqDetDatatset(args.data_path, args, train=True)
    collate_fn = partial(protein_collate_fn, alphabet=alphabet, args=args)
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
    dl_train = torch.utils.data.DataLoader(ds_train, sampler=sampler_train, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Start training for {args.epochs} epochs, saving to {args.output_dir}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)
        train_one_epoch(model, dl_train, optimizer, device, epoch, args)
        if epoch % args.eval_period == args.eval_period - 1:
            for dl_test in dls_test:
                evaluate(model, dl_test, device, args)
        if epoch % args.save_period == args.save_period - 1:
            ckpt_path = misc.save_model(args, epoch, model, model_without_ddp, optimizer, None)
            print(f'Saved checkpoint to {ckpt_path}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    metrics = {}
    for dl_test in dls_test:
        metrics.update(evaluate(model, dl_test, device, args))

    if misc.is_main_process():
        metrics['copypasta'] = ',,'.join([metrics[f'{dl.dataset.name}_copypasta'] for dl in dls_test])
        print(metrics)

    if not args.disable_wandb and misc.is_main_process():
        wandb.log({'copypasta': metrics['copypasta']})
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)