import torch
import torch.nn as nn

from KWStreamingSearch.CTC.ctc_streaming_search import CTCStreamingSearch
from KWStreamingSearch.Transducer.trans_streaming_search import TransStreamingSearch 
from KWStreamingSearch.fusion_strategy import FusionStrategy

class MFAStreamingSearch(nn.Module):
    def __init__(self, keyword_ints, ctc_blank_id: int, trans_blank_id: int, fusion_name: str, fusion_conf: dict):
        self.keyword_ints = keyword_ints

        self.ctc_search = CTCStreamingSearch(keyword_ints, ctc_blank_id)
        self.trans_search = TransStreamingSearch(keyword_ints, trans_blank_id)
        self.fusion_strategy = FusionStrategy(fusion_name, fusion_conf)

    def forward(self, ctc_posterior, trans_posterior):
        pass