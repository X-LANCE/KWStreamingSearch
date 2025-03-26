import torch
import torch.nn as nn

'''
    ctc streaming
    cdc streaming
    transducer streaming
    joint streaming
'''
class KWSBaseSearch(nn.Module):
    def __init__(self, blank):
        self.blank = blank

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        # logits_lens is not used for KWS-based search. We don't remove it for compatibility. (ASR decoding requires it)
        raise NotImplementedError

    def streaming_search(self, posteriors: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        raise NotImplementedError
