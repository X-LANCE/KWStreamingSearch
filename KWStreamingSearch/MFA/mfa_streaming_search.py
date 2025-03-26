import torch
from KWStreamingSearch.base import KWSBaseSearch
from KWStreamingSearch.CTC.ctc_streaming_search import CTCPsdStreamingSearch
from KWStreamingSearch.Transducer.trans_streaming_search import RNNTStreamingSearch, TDTStreamingSearch
from KWStreamingSearch.fusion_strategy import FusionStrategy

class MFSStreamingSearch(KWSBaseSearch):
    def __init__(self, ctc_blank: int, trans_blank: int, fusion_name: str, fusion_conf: dict):
        self.ctc_blank = ctc_blank
        self.ctc_search = CTCPsdStreamingSearch(self.ctc_blank)
        self.trans_blank = trans_blank
        self.trans_search = RNNTStreamingSearch(self.trans_blank)
        self.fusion_strategy = FusionStrategy(fusion_name, fusion_conf)

    def forward(self, ctc_posterior: torch.tensor, trans_posterior: torch.tensor, 
                targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        pass


class MFAStreamingSearch(KWSBaseSearch):
    def __init__(self, ctc_blank: int, trans_blank: int, fusion_name: str, fusion_conf: dict):
        self.ctc_blank = ctc_blank
        self.ctc_search = CTCPsdStreamingSearch(self.ctc_blank)
        self.trans_blank = trans_blank
        self.trans_search = TDTStreamingSearch(self.trans_blank)
        self.fusion_strategy = FusionStrategy(fusion_name, fusion_conf)

    def forward(self, ctc_posterior: torch.tensor, trans_posterior: torch.tensor, 
                targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        pass