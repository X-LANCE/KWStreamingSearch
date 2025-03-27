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

    def postprocessing(self, scores: list, total_tlist: list, bonus: float=3.5, timeout: int=100):
        # s_bonus
        scores = [_ + bonus if _ != 'placeholder' else 'placeholder' for _ in scores]

        # timeout
        for t in range(len(scores)):
            if scores[t] == 'placeholder':
                continue

            if total_tlist[t] > timeout:
                scores[t] = -1e35

        # normalization by path length
        normed_scores = [(scores[t] / total_tlist[t] if total_tlist[t] != 'placeholder' else 'placeholder') for t in range(len(scores))]
        normed_scores = [torch.Tensor([_]).to(self.device) if isinstance(_, float) else _ for _ in normed_scores]

        return normed_scores