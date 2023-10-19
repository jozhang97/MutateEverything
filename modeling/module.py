import math
import torch
import torch.nn as nn

from einops import rearrange

from modeling.utils import FFNLayer

######################################
# AA Expansion
######################################
class AAExpander(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.bb_dim = backbone.hdim
        self.head_dim = args.head_dim

        self.aa_expander = nn.Linear(self.bb_dim, 20*self.head_dim)
        self.aa_embed = nn.Embedding(20, self.head_dim)

    def forward(self, x, batch, pred):
        bb_feat = pred['bb_feat']
        B, L, _ = bb_feat.shape
        mut1_feat = self.aa_expander(bb_feat).view(B,L,20,self.head_dim)

        aa_embed = rearrange(self.aa_embed.weight, 'a d -> 1 1 a d')
        mut1_feat = mut1_feat + aa_embed  # B,L,20,D
        return {'mut1_feat': mut1_feat}

class AAExpanderWithBackbone(nn.Module):
    ## uses backbone to obtain aa_embed
    def __init__(self, args, backbone):
        super().__init__()
        self.bb_dim = backbone.hdim
        self.aa_embed_dim = backbone.get_aa_embed_dim()
        self.head_dim = args.head_dim

        self.aa_expander = nn.Linear(self.bb_dim, 20*self.head_dim)
        self.aatype_adapter = nn.Sequential(
            nn.LayerNorm(self.aa_embed_dim),
            nn.Linear(self.aa_embed_dim, self.head_dim),
            FFNLayer(self.head_dim, self.head_dim),
        )

    def forward(self, x, batch, pred):
        bb_feat = pred['bb_feat']
        B, L, _ = bb_feat.shape
        mut1_feat = self.aa_expander(bb_feat).view(B,L,20,self.head_dim)

        aa_embed = pred['aa_embed'].view(1, 1, 20, -1)
        aa_embed = self.aatype_adapter(aa_embed)
        mut1_feat = mut1_feat + aa_embed  # B,L,20,D
        return {'mut1_feat': mut1_feat}

def create_aa_expander(args, backbone):
    if args.aa_expand == 'scratch':
        return AAExpander(args, backbone)
    elif args.aa_expand == 'backbone':
        return AAExpanderWithBackbone(args, backbone)
    else:
        assert False, f'Unknown aa_expand: {args.aa_expand}'
    

######################################
# Single Decoders
######################################
class NaiveSingleDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head_dim = args.head_dim
        self.head = nn.Linear(self.head_dim, 1)
        
    def forward(self, x, batch, pred):
        mut1_ddg = self.head(pred['mut1_feat']).squeeze(-1)
        return {'mut1_ddg': mut1_ddg}

def create_single_decoder(args):
    return NaiveSingleDecoder(args)

######################################
# Multiple Decoders
######################################
class AdditiveMultiDecoder(nn.Module):
    ## Baseline that naively adds the single mutation ddgs
    def forward(self, x, batch, pred):
        mut1_ddg = pred['mut1_ddg']
        mut2_ddg = mut1_ddg[:,None,None,:,:] + mut1_ddg[:,:,:,None,None]
        return {'mut2_ddg': mut2_ddg}

class EpistaticMultiDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head_dim = args.head_dim
        self.chunk_size = 50
        self.test = args.test
        
        self.multi_adapter = FFNLayer(self.head_dim, self.head_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            FFNLayer(self.head_dim, self.head_dim),
            nn.Linear(self.head_dim, 1),
        )
    
    def forward(self, x, batch, pred):
        mut1_feat = self.multi_adapter(pred['mut1_feat'])
        B, L, A, D = mut1_feat.shape
        ret = {}

        do_chunk = not self.training and L > self.chunk_size
        if do_chunk:
            chk = self.chunk_size
            n_chunks_sqrt = math.ceil(L / chk)
            mut2_ddg = mut1_feat.new_ones(B, L, 20, L, 20) * 999
            for i in range(n_chunks_sqrt):
                for j in range(n_chunks_sqrt):
                    if not self.test and \
                        not batch['known_mask2'][:,i*chk:(i+1)*chk,:,j*chk:(j+1)*chk,:].any():
                        continue
                    mut1_feat_i = mut1_feat[:,i*chk:(i+1)*chk]
                    mut1_feat_j = mut1_feat[:,j*chk:(j+1)*chk]
                    mut2_feat_ij = mut1_feat_i[:,:,:,None,None,:] + mut1_feat_j[:,None,None,:,:,:]
                    mut2_ddg_ij = self.head(mut2_feat_ij).squeeze(-1)
                    mut2_ddg[:,i*chk:(i+1)*chk,:,j*chk:(j+1)*chk,:] = mut2_ddg_ij
        else:
            mut2_feat = mut1_feat[:,None,None,:,:,:] + mut1_feat[:,:,:,None,None,:] # B,L,20,L,20,D
            mut2_ddg = self.head(mut2_feat).squeeze(-1)

        mut1_ddg = batch['ddg_dense'] if self.training else pred['mut1_ddg']
        mut2_ddg = mut2_ddg + (mut1_ddg[:,None,None,:,:] + mut1_ddg[:,:,:,None,None])
        ret['mut2_ddg'] = mut2_ddg

        ## apply multi-mutation heads
        if batch.get('eval_list', False):
            ## gather mutant features
            x_muts = []
            single_ddgs_summed = []
            for b in range(B):
                for pos, to_type in zip(batch['mut_pos_list'][b], batch['mut_to_type_list'][b]):
                    pos_idx0 = [p-1 for p in pos]
                    x_mut = mut1_feat[b,pos_idx0,to_type].sum(0)
                    x_muts.append(x_mut)
                    single_ddgs_summed.append(mut1_ddg[b,pos_idx0,to_type].sum())
            x_muts = torch.stack(x_muts)
            single_ddgs_summed = torch.stack(single_ddgs_summed).squeeze(-1)

            ## decode ddg
            pr_ddgs = self.head(x_muts).squeeze(-1)
            pr_ddgs = pr_ddgs + single_ddgs_summed
            chunk_sizes = [len(x) for x in batch['mut_info_list']]
            ret['pr_ddgs_list'] = torch.tensor_split(pr_ddgs, chunk_sizes)

            ### reverse
            ## gather mutant features
            x_muts = []
            single_ddgs_summed = []
            for b in range(B):
                for pos, to_type in zip(batch['mut_pos_list'][b], batch['mut_fr_type_list'][b]):
                    pos_idx0 = [p-1 for p in pos]
                    x_mut = mut1_feat[b,pos_idx0,to_type].sum(0)
                    x_muts.append(x_mut)
                    single_ddgs_summed.append(mut1_ddg[b,pos_idx0,to_type].sum())
            x_muts = torch.stack(x_muts)
            single_ddgs_summed = torch.stack(single_ddgs_summed).squeeze(-1)

            ## decode ddg
            pr_ddgs = self.head(x_muts).squeeze(-1)
            pr_ddgs = pr_ddgs + single_ddgs_summed
            chunk_sizes = [len(x) for x in batch['mut_info_list']]
            ret['pr_ddgs_self_list'] = torch.tensor_split(pr_ddgs, chunk_sizes)
        return ret

def create_multi_decoder(args):
    if args.multi_dec == 'epistasis':
        return EpistaticMultiDecoder(args)
    elif args.multi_dec == 'additive':
        return AdditiveMultiDecoder()
    else:
        assert False, f'Unknown multi_dec: {args.multi_dec}'
