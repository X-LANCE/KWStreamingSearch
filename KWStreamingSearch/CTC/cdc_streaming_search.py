import torch
import torch.nn as nn

from KWStreamingSearch.base import KWSBaseSearch

class CDCStreamingSearch(KWSBaseSearch):
    def __init__(self, keyword_ints, blank_id):
        super().__init__(keyword_ints, blank_id)

    def forward(self, ictc_posterior, ctc_posterior):
        pass
