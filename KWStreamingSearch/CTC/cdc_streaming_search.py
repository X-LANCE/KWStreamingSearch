import torch

from KWStreamingSearch.base import KWSBaseSearch
from .ctc_streaming_search import CTCFsdStreamingSearch

class CDCStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0):
        """
        Paper:
            title: Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency 
            arxiv: https://arxiv.org/pdf/2412.12635
            IEEE: https://ieeexplore.ieee.org/document/10890010 
        """
        super().__init__()
        self.ctc_streaming_decode = CTCFsdStreamingSearch(blank=blank)

    def forward(
        self, inter_logits: torch.Tensor, fianl_logits: torch.Tensor,
        targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        # intermediate layer decoding
        _, inter_logalpha_tlist, _, _ \
            = self.cdc_streaming_decode(inter_logits, targets, logits_lens, target_lens)
        
        # final layer decoding
        forward_logprob, logalpha_tlist, start_tlist, total_tlist \
            = self.ctc_streaming_decode(fianl_logits, targets, logits_lens, target_lens)
        
        # cdc-based fusion
        # TODO

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist
