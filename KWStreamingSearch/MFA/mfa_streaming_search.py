import torch
from KWStreamingSearch.base_search import KWSBaseSearch
from KWStreamingSearch.CTC.ctc_streaming_search import CTCPsdStreamingSearch
from KWStreamingSearch.Transducer.trans_streaming_search import TDTStreamingSearch
from KWStreamingSearch.fusion_strategy import FusionStrategy

class MFAStreamingSearch(KWSBaseSearch):
    def __init__(self, ctc_blank: int, trans_blank: int, fusion_name: str, fusion_conf: dict):
        self.ctc_blank = ctc_blank
        self.ctc_search = CTCPsdStreamingSearch(self.ctc_blank)
        self.trans_blank = trans_blank
        self.trans_search = TDTStreamingSearch(self.trans_blank)
        self.fusion_strategy = FusionStrategy(fusion_name, fusion_conf)

    def forward(self, ctc_logits: torch.tensor, trans_logits: torch.tensor, 
                targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        # CTC decoding
        _, ctc_alpha_tlist, _, _ \
            = self.ctc_search(ctc_logits, targets, logits_lens, target_lens)
        
        # TDT decoding
        forward_logprob, trans_alpha_tlist, start_tlist, total_tlist \
            = self.trans_search(trans_logits, targets, logits_lens, target_lens)
        
        # fusion
        alpha_tlist = self.fusion_strategy(trans_alpha_tlist, ctc_alpha_tlist)

        return forward_logprob, alpha_tlist, start_tlist, total_tlist