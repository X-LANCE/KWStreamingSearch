import torch

from KWStreamingSearch.base import KWSBaseSearch
from KWStreamingSearch.CTC.ctc_streaming_search import CTCFsdStreamingSearch
from KWStreamingSearch.fusion_strategy import FusionStrategy

class CDCStreamingSearch(KWSBaseSearch):
    def __init__(self, cdc_conf: dict, blank: int = 0):
        """
        Paper:
            title: Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency 
            arxiv: https://arxiv.org/pdf/2412.12635
            IEEE: https://ieeexplore.ieee.org/document/10890010 
        """
        super().__init__()
        self.ctc_streaming_decode = CTCFsdStreamingSearch(blank=blank)
        self.fusion_strategy = FusionStrategy('cdc-zero', cdc_conf)

    def forward(
        self, inter_logits: torch.Tensor, final_logits: torch.Tensor,
        targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        # intermediate layer decoding
        _, inter_logalpha_tlist, _, _ \
            = self.ctc_streaming_decode(inter_logits, targets, logits_lens, target_lens)
        
        # final layer decoding
        forward_logprob, logalpha_tlist, start_tlist, total_tlist \
            = self.ctc_streaming_decode(final_logits, targets, logits_lens, target_lens)
        
        # cdc-based fusion
        logalpha_tlist = self.fusion_strategy(inter_logalpha_tlist, logalpha_tlist)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist
