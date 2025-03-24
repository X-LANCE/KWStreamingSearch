import torch
import torch.nn as nn

from KWStreamingSearch.base import KWSBaseSearch

class CTCStreamingSearch(KWSBaseSearch):
    def __init__(self, keyword_ints, blank_id, blank_filter_threshold=1):
        super().__init__(keyword_ints, blank_id)
        self.blank_filter_threshold = blank_filter_threshold

    def forward(self, posterior: torch.Tensor):
        pass

