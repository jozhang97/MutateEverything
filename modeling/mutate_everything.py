import torch.nn as nn
from modeling.backbone import create_backbone
from modeling.module import create_aa_expander, create_single_decoder, create_multi_decoder

class MutateEverything(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = create_backbone(args)
        self.aa_expansion = create_aa_expander(args, self.backbone)
        self.single_decoder = create_single_decoder(args)
        self.multi_decoder = create_multi_decoder(args)

    def forward(self, x, batch):
        pred = {}
        pred.update(self.backbone(x, batch))
        pred.update(self.aa_expansion(x, batch, pred))
        pred.update(self.single_decoder(x, batch, pred))
        pred.update(self.multi_decoder(x, batch, pred))
        return pred
