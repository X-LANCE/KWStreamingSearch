import torch
import torch.nn as nn

'''
    ctc streaming
    cdc streaming
    transducer streaming
    joint streaming
'''
class KWSBaseSearch(nn.Module):
    def __init__(self, keyword_ints, blank_id):
        self.keyword_ints = keyword_ints
        self.blank_id = blank_id

    def forward(self, posterior: torch.Tensor):
        raise NotImplementedError
