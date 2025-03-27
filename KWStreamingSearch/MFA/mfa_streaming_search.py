import torch
from KWStreamingSearch.base_search import KWSBaseSearch
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
        # RNN-T decoding
        _, ctc_logalpha_tlist, _, _ \
            = self.ctc_search(ctc_posterior, targets, logits_lens, target_lens)
        
        # CTC decoding
        forward_logprob, trans_logalpha_tlist, start_tlist, total_tlist \
            = self.trans_search(trans_posterior, targets, logits_lens, target_lens)
        
        # fusion
        logalpha_tlist = self.fusion_strategy(trans_logalpha_tlist, ctc_logalpha_tlist)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist


class MFAStreamingSearch(KWSBaseSearch):
    def __init__(self, ctc_blank: int, trans_blank: int, fusion_name: str, fusion_conf: dict):
        self.ctc_blank = ctc_blank
        self.ctc_search = CTCPsdStreamingSearch(self.ctc_blank)
        self.trans_blank = trans_blank
        self.trans_search = TDTStreamingSearch(self.trans_blank)
        self.fusion_strategy = FusionStrategy(fusion_name, fusion_conf)

    def forward(self, ctc_posterior: torch.tensor, trans_posterior: torch.tensor, 
                targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        # TDT decoding
        _, ctc_logalpha_tlist, _, _ \
            = self.ctc_search(ctc_posterior, targets, logits_lens, target_lens)
        
        # CTC decoding
        forward_logprob, trans_logalpha_tlist, start_tlist, total_tlist \
            = self.trans_search(trans_posterior, targets, logits_lens, target_lens)
        
        # fusion
        logalpha_tlist = self.fusion_strategy(trans_logalpha_tlist, ctc_logalpha_tlist)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist