import datetime
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO

import torch

import misc
from modeling.mutate_everything import MutateEverything
from data import one_to_three, one_letters

def get_args_parser():
    parser = argparse.ArgumentParser('Output ddgs for all single and double mutations')

    # input params
    parser.add_argument('--name', type=str, help='name to save under')
    parser.add_argument('--seq', type=str, help='raw sequence or fasta file')
    parser.add_argument('--msa_dir', type=str, help='directory with 1 or more a3m files')
    parser.add_argument('--output_dir', type=Path, default='logs/debug')
    parser.add_argument('--seed', type=int, default=0)

    # model params
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--aa_expand', default='backbone', help='scratch|backbone')
    parser.add_argument('--single_dec', default='naive', help='naive')
    parser.add_argument('--multi_dec', default='epistasis', help='additive|epistasis')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--backbone', default='esm2_t33_650M_UR50D', help='af|esm2_t33_650M_UR50D')
    parser.add_argument('--finetune_backbone', type=str, default='models/finetuning_ptm_2.pt')
    parser.add_argument('--freeze_at', default=0, help='freeze backbone up to layer X')
    parser.add_argument('--device', default='cuda')

    # af params
    parser.add_argument('--n_msa_seqs', type=int, default=128)
    parser.add_argument('--n_extra_msa_seqs', type=int, default=1024)
    parser.add_argument('--af_extract_feat', type=str, default='both',
        help='which features to use from AF: both|evo|struct')
    return parser

@torch.no_grad()
def forward_esm(model, alphabet, args):
    device = torch.device(args.device)

    ## tokenize sequences
    seqs = [('1', load_seq(args.seq))]
    batch_converter = alphabet.get_batch_converter()
    _, _, x = batch_converter(seqs)
    x = x.to(device)

    ## forward model
    model.to(device)
    pred = model(x, {'seqs': [load_seq(args.seq)]})
    return pred

@torch.no_grad()
def forward_af(model, args):
    device = torch.device(args.device)

    from openfold.config import model_config
    from openfold.data import feature_pipeline, data_pipeline

    ## configs
    config = model_config('finetuning', train=True)
    config.data.train.max_extra_msa = 1024
    config.data.predict.max_extra_msa = 1024
    config.data.train.max_msa_clusters = 128
    config.data.predict.max_msa_clusters = 128

    ## prepare inputs
    data_processor = data_pipeline.DataPipeline(None)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    feature_dict = data_processor.process_fasta(args.seq, args.msa_dir)
    af_inputs = feature_processor.process_features(
        feature_dict,
        mode='predict',
    )
    x = [{k: v.to(device) for k, v in af_inputs.items()}]

    ## forward model
    model.to(device)
    pred = model(x, {'seqs': [load_seq(args.seq)]})
    return pred

def load_seq(seq):
    if '.fasta' in seq:
        for record in SeqIO.parse(seq, 'fasta'):
            seq = str(record.seq)
    return seq

def main(args):
    print('WARNING: We observe a cysteine stabilization bias when examining DMS predictions (cysteine is often predicted to be the most stabilizing substitution). We are unsure if this is an artifact from the training data but attempts to fix this bias lead to worse metrics on the test set. Use cysteine predictions with caution.')
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## prepare model
    model = MutateEverything(args)
    misc.load_model(args, model, None, None)
    model.eval()

    ## logging
    print(f'Start testing')
    start_time = time.time()

    ## forward model
    seq = load_seq(args.seq)
    if args.backbone == 'af':
        pred = forward_af(model, args)
    elif 'esm' in args.backbone:
        _, alphabet = model.backbone.get_alphabet()
        pred = forward_esm(model, alphabet, args)
    mut1_ddg = pred['mut1_ddg'][0].cpu()
    mut2_ddg = pred['mut2_ddg'][0].cpu()

    ## save single predictions
    rows = []
    for l in range(len(seq)):
        mut1_ddg_l = mut1_ddg[l]
        muts = {one_to_three[k]: v for k, v in zip(one_letters, mut1_ddg_l)}
        muts = {f'pr{k}': f'{muts[k].item():.04f}' for k in sorted(muts)}
        rows.append({
            'seq_num': l + 1,
            'wtAA': one_to_three[seq[l]],
            'predAA': one_to_three[one_letters[mut1_ddg_l.argmin()]],
            'pred_ddG': f'{mut1_ddg_l.min().item():.04f}',
            'stable_mut_count': (mut1_ddg_l < -0.5).sum().item(),
            'neutral_mut_count': ((-0.5 < mut1_ddg_l) & (mut1_ddg_l < 0.5)).sum().item(),
            'destable_mut_count': (mut1_ddg_l > 0.5).sum().item(),
            **muts,
            'seq': seq,
        })
    df = pd.DataFrame.from_dict(rows)
    fp = args.output_dir / f'{args.name}_single.csv'
    print(f'Writing pred dms to {fp}')
    df.to_csv(fp, index=False)

    ## save double predictions
    stbl2 = mut2_ddg < -0.5
    p1s, a1s, p2s, a2s = stbl2.nonzero(as_tuple=True)
    cond = (p1s < p2s) & (a1s < a2s)  # only upper triangle
    p1s = p1s[cond]
    a1s = a1s[cond]
    p2s = p2s[cond]
    a2s = a2s[cond]

    muts = []
    for p1, a1, p2, a2 in zip(p1s, a1s, p2s, a2s):
        wt1 = seq[p1]
        wt2 = seq[p2]
        if wt1 == one_letters[a1] or wt2 == one_letters[a2]:
            continue
        m1 = f'{wt1}{p1+1}{one_letters[a1]}'
        m2 = f'{wt2}{p2+1}{one_letters[a2]}'
        ddg = mut2_ddg[p1,a1,p2,a2].item()
        muts.append((m1, m2, ddg))
    df2 = pd.DataFrame(muts, columns=['mut1', 'mut2', 'ddG'])
    df2 = df2.sort_values('ddG')
    fp = args.output_dir / f'{args.name}_double.csv'
    print(f'Writing stabilizing doubles to {fp}')
    df2.to_csv(fp, index=False)

    ## logging
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.finetune = None
    args.test = True
    args.eval = True
    print(args)
    main(args)
