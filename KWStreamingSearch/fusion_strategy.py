import torch
import torch.nn as nn

class FusionStrategy(nn.Module):
    def __init__(self, fusion_name, fusion_conf):
        super().__init__()
        self.fusion_name = fusion_name
        self.fusion_conf = fusion_conf

        self._call_dict = {
            'trans-dom': self.trans_dom,
            'cdc-zero': self.cdc_zero,
        }
    
    def cdc_zero(self, trans_scores, ctc_scores):
        pass

    def trans_dom(self, trans_scores, ctc_scores):
        pass