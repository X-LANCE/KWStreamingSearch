import torch
import torch.nn as nn

from KWStreamingSearch.base import KWSBaseSearch

class TransStreamingSearch(KWSBaseSearch):
    def __init__(self, keyword_ints, blank_id, max_duration):
        super().__init__(keyword_ints, blank_id)
        self.max_duration = max_duration

    def forward(self, posterior: torch.Tensor):
        pass
