from pathlib import Path

import numpy as np
import pandas as pd

import torch

UNKNOWN_VALUE = 999
d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
one_to_three = {v: k for k, v in d.items()}
three_letters = np.array(sorted(list(d.keys())))
three_letters_list = list(three_letters)
aa = sorted(d.values())
one_letters = np.array(aa)
one_letters_list = list(aa)

class SeqDetDatatset(torch.utils.data.Dataset):
    def __init__(self, fp, args, train=True):
        """
        data_dir = fp.parent.parent
        data_dir/
            mutations/
                fp.name
            fasta/
                pdb_id/
                    pdb_id.fasta
            msa/
                pdb_id/
                    bfd.a3m
                    ...
        """
        self.train = train
        self.args = args
        self.name = Path(fp).stem
        self.data_dir = Path(fp).parent.parent
        self.msa_dir = self.data_dir / 'msa'
        self.fasta_dir = self.data_dir / 'fasta'
        self.max_context_length = args.max_context_length

        # convert df of mutations to df of pdb
        df = pd.read_csv(fp, low_memory=False)
        df = df[(~df.mut_info.isna()) & (~df.wt_seq.isna())]
        df = df[~df.ddg.isna()]

        df['seq'] = df['wt_seq']
        detdf = df.groupby('pdb_id', as_index=False).head(1)

        ## remove proteins with multiple wt_seq
        n_seq_per_pdb = df.groupby('pdb_id').wt_seq.nunique()
        if not (n_seq_per_pdb == 1).all():
            pdb_with_one_seq = n_seq_per_pdb[n_seq_per_pdb == 1].index
            pdb_with_few_seq = n_seq_per_pdb[n_seq_per_pdb != 1].index
            detdf = detdf[detdf.pdb_id.isin(pdb_with_one_seq)]
            df = df[df.pdb_id.isin(pdb_with_one_seq)]
            print(f'WARNING: Found multiple wt_seq, Removing {pdb_with_few_seq}')

        self.mutdf = df
        self.df = detdf
        print(f'Loaded {len(self.df)} pdbs ({len(self.mutdf)} mutations) from {fp}')

        ## Prepare AF pipeline
        if self.args.backbone == 'af':
            from openfold.config import model_config
            from openfold.data import feature_pipeline, data_pipeline

            config = model_config('finetuning', train=True)
            config.model.template.enabled = False
            config.model.evoformer_stack.tune_chunk_size = False
            config.data.train.max_extra_msa = 1024
            config.data.predict.max_extra_msa = 1024
            config.data.train.max_msa_clusters = 128
            config.data.predict.max_msa_clusters = 128

            self.data_processor = data_pipeline.DataPipeline(None)
            self.feature_processor = feature_pipeline.FeaturePipeline(config.data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        ret = combine_protein_mutations(self.mutdf[self.mutdf.pdb_id == row.pdb_id], args=self.args, train=self.train)
        if self.args.backbone == 'af':
            pdb_id = ret.pdb_id
            fasta_fp = self.fasta_dir / f'{pdb_id}/{pdb_id}.fasta'
            if not fasta_fp.exists():
                pdb_id = pdb_id[:4]
            try:
                feature_dict = self.data_processor.process_fasta(
                    self.fasta_dir / f'{pdb_id}/{pdb_id}.fasta',
                    self.msa_dir / pdb_id
                )
            except ValueError as e:
                print(f'likely msa entry for {pdb_id} is not the right length at {self.msa_dir / pdb_id}\n{e}')
                raise e
            processed_feature_dict = self.feature_processor.process_features(
                feature_dict,
                mode='predict',
            )
            ret['af_inputs'] = processed_feature_dict

        ## trim to max_context_length if long and muts are not too far
        context_length = len(ret.seq)
        if context_length > self.max_context_length:
            ## get all positions
            mut_infos = self.mutdf[self.mutdf.pdb_id == ret.pdb_id].mut_info
            pos_df = mut_infos.apply(lambda x: [int(xx[1:-1]) for xx in x.split(':')])
            pos_list = set([xx for x in pos_df for xx in x])

            ## get start and end pos
            min_pos, max_pos = min(pos_list)-1, max(pos_list)+1
            assert max_pos - min_pos <= self.max_context_length, 'muts too far away'
            mid_pos = (min_pos + max_pos) // 2
            half_context = self.max_context_length // 2
            start_pos, end_pos = mid_pos - half_context, mid_pos + half_context
            if start_pos < 0:
                start_pos = 0
                end_pos = self.max_context_length

            ## trim af_inputs
            if 'af_inputs' in ret:
                processed_feature_dict['seq_length'] = self.max_context_length + processed_feature_dict['seq_length'] * 0
                for k, v in processed_feature_dict.items():
                    if v.shape[0] == context_length:
                        processed_feature_dict[k] = v[start_pos: end_pos]
                    if len(v.shape) > 1 and v.shape[1] == context_length:
                        processed_feature_dict[k] = v[:, start_pos: end_pos]
                ret['af_inputs'] = processed_feature_dict

            ## trim inputs and labels
            ret['seq'] = ret['seq'][start_pos: end_pos]
            ret['ddg_dense'] = ret['ddg_dense'][start_pos: end_pos]
            ret['ddg_dense2'] = ret['ddg_dense2'][start_pos: end_pos, :, start_pos: end_pos, :]
            assert len(ret['ddg_list']) == 0
        return ret

def combine_protein_mutations(df: pd.DataFrame, args=None, train=False):
    """
    Args:
        df: dataframe of mutations for a single pdb
            contains ['pdb_id', 'wt_seq', 'mut_info', 'ddg']
    Returns:
        seq: wild-type sequence
        ddg_dense: Lx20 matrix
            experimental ddG for mutant amino acid
            0 for wild-type amino acid
            UNKNOWN_VALUE for missing amino acid
    """
    if df.wt_seq.nunique() > 1:
        print(f'WARNING: {df.pdb_id.unique()} contains multiple sequences, skipping')
        return
    seq = df.wt_seq.iloc[0]

    self_mut_value = 0 if train else UNKNOWN_VALUE
    ret = {}
    ddg_dense = np.ones((len(seq), 20), dtype=np.float32) * UNKNOWN_VALUE
    ddg_dense2 = np.ones((len(seq), 20, len(seq), 20), dtype=np.float32) * UNKNOWN_VALUE
    for _, row in df.iterrows():
        ddg = row['ddg']
        if row.mut_info.count(':') == 1:
            fr1, pos1, to1 = row.fr1, int(row.pos1), row.to1
            fr2, pos2, to2 = row.fr2, int(row.pos2), row.to2
            assert seq[pos1 - 1] == fr1, f'from_aa ({seq[pos1 - 1]}) at pos {pos1} does not match mut_info {row.mut_info}'
            assert seq[pos2 - 1] == fr2, f'from_aa ({seq[pos2 - 1]}) at pos {pos2} does not match mut_info {row.mut_info}'
            assert pos1 < pos2, f'invalid mutation order {row.mut_info}'

            fr1_aa_idx = aa.index(fr1)
            fr2_aa_idx = aa.index(fr2)
            to1_aa_idx = aa.index(to1)
            to2_aa_idx = aa.index(to2)

            ddg_dense2[pos1 - 1, fr1_aa_idx, pos2 - 1, fr2_aa_idx] = self_mut_value
            ddg_dense2[pos1 - 1, to1_aa_idx, pos2 - 1, to2_aa_idx] = ddg + 1e-6 * torch.randn(1).item()
        elif ':' not in row.mut_info:
            mut_info = row['mut_info']

            fr, pos, to = mut_info[0], int(mut_info[1:-1]), mut_info[-1]  # mut_info always index by 1
            assert seq[pos - 1] == fr, f'from_aa ({seq[pos - 1]}) at pos {pos} does not match mut_info {mut_info}. Surrounding sequence: {seq[pos-6:pos+6]}(are you using struct pos?)'
            fr_aa_idx = aa.index(fr)
            to_aa_idx = aa.index(to)

            ddg_dense[pos - 1, fr_aa_idx] = self_mut_value
            ddg_dense[pos - 1, to_aa_idx] = ddg + 1e-6 * torch.randn(1).item()

    ret = {'seq': seq, 'ddg_dense': ddg_dense, 'ddg_dense2': ddg_dense2}
    ret['pdb_id'] = df.pdb_id.unique().item()

    one_letter_idx = np.array([one_letters_list.index(aa) if aa in one_letters_list else 0 for aa in seq])
    ret['one_letter_idx'] = one_letter_idx

    n_subs_in_mut = df.mut_info.str.count(':') + 1
    if 'mut_pos_list' in df.columns:
        df_multi = df[n_subs_in_mut > 2]
        ret['mut_pos_list'] = df_multi['mut_pos_list'].apply(lambda x: [int(xx) for xx in eval(x)]).tolist()
        ret['mut_fr_type_list'] = df_multi['mut_fr_list'].apply(lambda x: [one_letters_list.index(xx) for xx in eval(x)]).tolist()
        ret['mut_to_type_list'] = df_multi['mut_to_list'].apply(lambda x: [one_letters_list.index(xx) for xx in eval(x)]).tolist()
        ret['mut_info_list'] = df_multi['mut_info'].tolist()
        ret['ddg_list'] = df_multi['ddg'].tolist()
    return pd.Series(ret)

def protein_collate_fn(batch, alphabet, args):
    ## convert sequences into tokens https://github.com/facebookresearch/esm/blob/0b59d87ebef95948c735b1f7aad463dc6dfa991b/esm/data.py#L253
    # <SOS> SEQ <EOS> <PAD>
    inputs = {}
    alphabet = alphabet[1]

    if args.backbone == 'af':
        inputs['af_inputs'] = [pdb.af_inputs for pdb in batch]
    else:
        batch_converter = alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter([(pdb.pdb_id, pdb.seq) for pdb in batch])
        inputs['tokens'] = batch_tokens

    if any(['mut_info_list' in pdb for pdb in batch]):
        inputs.update({
            'mut_info_list': [pdb.mut_info_list for pdb in batch],
            'mut_pos_list': [pdb.mut_pos_list for pdb in batch],
            'mut_fr_type_list': [pdb.mut_fr_type_list for pdb in batch],
            'mut_to_type_list': [pdb.mut_to_type_list for pdb in batch],
            'ddg_list': [pdb.ddg_list for pdb in batch],
        })
    one_letter_idx = [pdb.one_letter_idx for pdb in batch]
    one_letter_idx = torch.tensor(np.array(one_letter_idx), dtype=torch.long)

    if args.test:
        return {
            'pdb_ids': [pdb.pdb_id for pdb in batch],
            'seqs': [pdb.seq for pdb in batch],
            'one_letter_idx': one_letter_idx,
            **inputs,
        }

    ## batch ddg masks and handle padding
    ddg_dense = [torch.tensor(pdb.ddg_dense, dtype=torch.float32) for pdb in batch]
    max_seq_len = max([t.shape[0] for t in ddg_dense])
    padded_ddg_dense = [torch.cat([
        t,
        t.new_ones(max_seq_len - t.shape[0], *t.shape[1:]) * UNKNOWN_VALUE,
    ], dim=0) for t in ddg_dense]
    padded_ddg_dense = torch.stack(padded_ddg_dense, dim=0)

    padded_ddg_dense2 = padded_ddg_dense.new_ones(len(batch), max_seq_len, 20, max_seq_len, 20) * UNKNOWN_VALUE
    for b, pdb in enumerate(batch):
        ddg_dense2 = torch.tensor(pdb.ddg_dense2, dtype=torch.float32)
        l = ddg_dense2.shape[0]
        padded_ddg_dense2[b,:l,:,:l,:] = ddg_dense2

    return {
        'ddg_dense': padded_ddg_dense,
        'ddg_dense2': padded_ddg_dense2,
        'one_letter_idx': one_letter_idx,
        'pdb_ids': [pdb.pdb_id for pdb in batch],
        'seqs': [pdb.seq for pdb in batch],
        **inputs,
    }

def get_dense_mut_infos(seq: str):
    # returns L*20 list of all single point mut_infos
    mut_infos = []
    L = len(seq)
    for pos in range(L):
        muts_for_i = [f'{seq[pos]}{pos+1}{to}' for to in aa]
        mut_infos.extend(muts_for_i)
    return mut_infos

def get_dense_double_mut_infos(seq):
    # returns L*20*L*20 list of all double mut_infos
    L = len(seq)
    mut_infos = get_dense_mut_infos(seq)
    mut_infos2 = []
    valid_mask = []
    for m1 in mut_infos:
        for m2 in mut_infos:
            mut_infos2.append(f'{m1}:{m2}')
            pos1 = int(m1[1:-1])
            pos2 = int(m2[1:-1])
            valid_mask.append(pos1 < pos2)
    return np.array(mut_infos2), np.array(valid_mask)
