import torch
import torch.nn as nn

import esm

from modeling.utils import FFNLayer
from data import one_letters

class AlphaFoldBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hdim = 384
        self._prepare_alphafold()

        ## prepare feature extractor
        self.af_extract_feat = args.af_extract_feat
        assert self.af_extract_feat in ['evo', 'struct', 'both']
        self.ln = nn.LayerNorm(self.hdim)
        if self.af_extract_feat in ['evo', 'both']:
            self.bb_adapter1 = FFNLayer(self.hdim, self.hdim)
        if self.af_extract_feat in ['struct', 'both']:
            self.bb_adapter2 = FFNLayer(self.hdim, self.hdim)

        ## prepare penultimate aa_type embeddings
        self.aa_expand = args.aa_expand
        if self.aa_expand == 'backbone':
            from openfold.np.residue_constants import restype_order
            self.aa_embed = self.backbone.aux_heads.masked_msa.linear.weight.requires_grad_(True)
            self.openfold_to_our_aatype = [restype_order[aa] for aa in one_letters]
            self.ln_head = nn.LayerNorm(self.get_aa_embed_dim())

    def get_aa_embed_dim(self):
        return 256

    def get_aa_embed(self):
        return self.ln_head(self.aa_embed[self.openfold_to_our_aatype])

    def _prepare_alphafold(self):
        from openfold.config import model_config
        from openfold.model.model import AlphaFold

        ## Hard-coded configs
        config = model_config('finetuning', train=not self.args.eval)
        config.globals.use_flash = False
        config.model.template.enabled = False
        config.model.evoformer_stack.tune_chunk_size = self.args.eval
        config.data.train.max_extra_msa = self.args.n_extra_msa_seqs
        config.data.predict.max_extra_msa = self.args.n_extra_msa_seqs
        config.data.train.max_msa_clusters = self.args.n_msa_seqs
        config.data.predict.max_msa_clusters = self.args.n_msa_seqs

        ## Prepare backbone model
        self.backbone = AlphaFold(config)
        print(f'Loading AF from {self.args.finetune_backbone}')
        ckpt = torch.load(self.args.finetune_backbone, map_location='cpu')
        self.backbone.load_state_dict(ckpt, strict=False)
        if self.args.freeze_at > 0:
            raise NotImplementedError

        ## avoid distributed issues since we do not train these
        self.backbone.aux_heads.requires_grad_(False)
        self.backbone.structure_module.angle_resnet.requires_grad_(False)

    def get_alphabet(self):
        return 'af', None

    def forward(self, x, batch):
        ## run AF
        assert len(x) == 1, 'only support bs=1'
        af_out = self.backbone(x[0])

        ## extract AF features
        x = 0
        if self.af_extract_feat in ['evo', 'both']:
            af_feats1 = af_out['single_post_evoformer']
            x = x + self.bb_adapter1(self.ln(af_feats1))
        if self.af_extract_feat in ['struct', 'both']:
            af_feats2 = af_out['single_post_structure_module']
            x = x + self.bb_adapter2(self.ln(af_feats2))
        x = x[None]  # B,L,D

        ret = {
            'bb_feat': x,
            'af_pair': af_out['pair'][None],
            'af_single_post_evoformer': af_out['single_post_evoformer'][None],
            'af_single_post_structure_module': af_out['single_post_structure_module'][None],
        }
        if self.aa_expand:
            ret['aa_embed'] = self.get_aa_embed()[None]
        return ret


class ESMBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.alphabet = getattr(esm.pretrained, args.backbone)()
        self.num_layers = len(self.backbone.layers)
        self.hdim = self.backbone.lm_head.dense.weight.shape[1]

        if args.freeze_at > 0:
            self.backbone.embed_tokens.requires_grad_(False)
            for i, layer in enumerate(self.backbone.layers):
                if i < args.freeze_at:
                    layer.requires_grad_(False)

        ## prepare feature extractor
        self.ln = nn.LayerNorm(self.hdim)
        self.bb_adapter = FFNLayer(self.hdim, self.hdim)

        ## prepare penultimate aa_type embeddings
        self.aa_expand = args.aa_expand
        if self.aa_expand == 'backbone':
            self.aa_embed = self.backbone.lm_head.weight.requires_grad_(True)
            self.esm_to_our_aatype = [self.alphabet.get_idx(aa) for aa in one_letters]
            self.ln_head = nn.LayerNorm(self.get_aa_embed_dim())

        ## avoid distributed issues
        self.backbone.lm_head.requires_grad_(False)
        self.backbone.contact_head.requires_grad_(False)

    def get_aa_embed_dim(self):
        return self.aa_embed.shape[1]

    def get_aa_embed(self):
        return self.ln_head(self.aa_embed[self.esm_to_our_aatype])

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch):
        x = self.backbone(x, repr_layers=[self.num_layers])['representations'][self.num_layers]
        if len(x.shape) == 4:  # remove MSAs
            x = x[:,0]
        x = self.bb_adapter(self.ln(x))
        x = x[:, 1:-1]  # remove SOS and EOS tokens
        ret = {'bb_feat': x}
        if self.aa_expand == 'backbone':
            ret['aa_embed'] = self.get_aa_embed()
        return ret


def create_backbone(args):
    if args.backbone == 'af':
        backbone = AlphaFoldBackbone(args)
    elif 'esm' in args.backbone.lower():
        backbone = ESMBackbone(args)
    else:
        assert False, f'unknown backbone type {args.backbone}'
    return backbone
