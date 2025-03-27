import torch
import torch.nn as nn

from KWStreamingSearch.fusion_strategy import PH

'''
    ctc streaming
    cdc streaming
    transducer streaming
    joint streaming
'''
# class KWSBaseSearch(nn.Module):
class KWSBaseSearch:
    def __init__(self, blank):
        self.blank = blank

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        # logits_lens is not used for KWS-based search. We don't remove it for compatibility. (ASR decoding requires it)
        raise NotImplementedError

    def streaming_search(self, posteriors: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def postprocessing(self, logscores: list, total_tlist: list, bonus: float=3.5, timeout: int=100):
        # s_bonus
        logscores = [_ + bonus if _ != PH else PH for _ in logscores]
        T = len(logscores)

        # timeout
        for t in range(T):
            if logscores[t] == PH:
                continue

            if total_tlist[t] > timeout:
                logscores[t] = -1e35

        # normalization by path length
        normed_logscores = [(logscores[t] / total_tlist[t] if total_tlist[t] != PH else PH) for t in range(T)]
        normed_scores = [torch.Tensor([_]).exp().to(self.device) if isinstance(_, float) else _ for _ in normed_logscores]

        return normed_scores