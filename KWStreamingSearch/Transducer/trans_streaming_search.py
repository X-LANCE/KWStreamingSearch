import torch

from typing import List
from KWStreamingSearch.base import KWSBaseSearch

class RNNTStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0):
        """
        Paper:
            title: TDT-KWS: Fast And Accurate Keyword Spotting Using Token-and-duration Transducer
            arxiv: https://arxiv.org/abs/2403.13332
            IEEE: https://ieeexplore.ieee.org/document/10446909
        """
        super().__init__(blank)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        """
        Args:
            logits: (torch.Tensor): B x T x U x D; B = 1, D = len(vocab_size) + 1(blank)
                                    When calculating logits, the input of the predictor network 
                                    is [blank, kwd_phoneme1, kwd_phoneme2, ..., kwd_phonemeN]. 
                                    Details can refer to the paper TDT-KWS: https://arxiv.org/abs/2403.13332
            targets: (torch.Tensor): B x U; B = 1
            logit_lens: (torch.Tensor): B; B = 1
            target_lens: (torch.Tensor): B; B = 1

        Returns:
            The maenings of the return values can refer to the comments in the CTCFsdStreamingSearch.forward.
        """
        posteriors = torch.log_softmax(logits, dim=-1)
        
        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.streaming_search(posteriors, targets, logits_lens, target_lens)
        
        return forward_logprob, logalpha_tlist, start_tlist, total_tlist

    def streaming_search(
        self, posteriors: torch.Tensor, targets: torch.Tensor,  logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        B, T, U, _ = posteriors.shape
        assert B == 1, f"decoding batch size must be 1, get B = {B}"

        log_alpha = torch.zeros(B, T, U) - 1000
        log_alpha = log_alpha.to(posteriors.device)
        start_alpha = torch.zeros(B, T, U)
        total_alpha = torch.zeros(B, T, U)
       
        log_alpha_each_t = ["placeholder" for _ in range(T)]
        start_alpha_each_t = ["placeholder" for _ in range(T)]
        total_alpha_each_t = ["placeholder" for _ in range(T)]
        
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[0, t, u] = 0.0
                        start_alpha[0, t, u] = 0
                        total_alpha[0, t, u] = 1
                    else:
                        log_alpha[0, t, u] = 0.0
                        start_alpha[0, t, u] = t 
                        total_alpha[0, t, u] = 1
                else:
                    if t == 0:
                        gathered = torch.gather(
                            posteriors[0, t, u - 1, :], dim=0, index=targets[0, u - 1].view(-1).type(torch.int64)).squeeze() # shape: [1] -> scalar
                        log_alpha[0, t, u] = log_alpha[0, t, u - 1] + gathered
                        start_alpha[0, t, u] = 0
                        total_alpha[0, t, u] = u + 1
                    else:
                        from_left = log_alpha[0, t - 1, u] + posteriors[0, t - 1, u, self.blank]

                        down_gathered = torch.gather(
                            posteriors[0, t, u - 1], dim=0, index=targets[0, u - 1].view(-1).type(torch.int64)).squeeze()
                        from_down = log_alpha[0, t, u - 1] + down_gathered

                        if from_left >= from_down:
                            log_alpha[0, t, u] = from_left
                            start_alpha[0, t, u] = start_alpha[0, t-1, u]
                            total_alpha[0, t, u] = total_alpha[0, t-1, u] + 1
                        else:
                            log_alpha[0, t, u] = from_down
                            start_alpha[0, t, u] = start_alpha[0, t, u-1]
                            total_alpha[0, t, u] = total_alpha[0, t, u-1] + 1

            out_log_alpha = log_alpha[0, t, target_lens[0]] + posteriors[0, t, target_lens[0], self.blank] # + reward
            out_start_alpha = start_alpha[0, t, target_lens[0]]
            out_total_alpha = total_alpha[0, t, target_lens[0]] + 1 # the last blank transition
            
            log_prob = out_log_alpha.item()
            log_alpha_each_t[t] = out_log_alpha
            start_alpha_each_t[t] = out_start_alpha.int().item()
            total_alpha_each_t[t] = out_total_alpha.int().item()

        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t

class TDTStreamingSearch(KWSBaseSearch):
    def __init__(self, durations: List[int], blank: int = 0):
        """ KWS streaming search for Token-and-Duration (TDT) Transducer.

        Args:
            durations (List[int]): All duration options for the TDT. 
                                   E.g., if D_{max} for TDT is 4, durations = [0, 1, 2, 3, 4].
            blank (int, optional): Blank id for TDT. Defaults to 0.
        """
        super().__init__(blank)
        self.durations = durations
        self.n_durations = len(durations)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor, t2skip
    ):
        """
        Args:
            logits: (torch.Tensor): B x T x U x D; B = 1, D = len(vocab_size) + 1(blank) + len(durations)
                                    When calculating logits, the input of the predictor network 
                                    is [blank, kwd_phoneme1, kwd_phoneme2, ..., kwd_phonemeN]. 
                                    Details can refer to the paper TDT-KWS: https://arxiv.org/abs/2403.13332
            targets: (torch.Tensor): B x U; B = 1
            logit_lens: (torch.Tensor): B; B = 1
            target_lens: (torch.Tensor): B; B = 1
            t2skip (List[int]): The skip frames at each frame t. The value is between 0 to D_{max}. It is infered from the greedy search of TDT.
                                E.g., if D_{max} for TDT is 2, the total length of the speech is 10 frames,
                                t2skip maybe equal to [0, 2, 0, 0, 1, 0, 2, 0, 0, 1]. In indicates at each frame t, how many frames should be skipped.
                                We don't provide this part of code as greedy search is heavily coupled to the model implementation. 
                                But it's very simple, as just can be derived from the Transudcer-based greedy search. 
        Returns:
            The maenings of the return values can refer to the comments in the CTCFsdStreamingSearch.forward.
        """
        target_posteriors = logits[:, :, :, :-self.n_durations]
        target_posteriors = torch.log_softmax(target_posteriors, dim=-1)

        #duration_logits = logits[:, :, :, -self.n_durations:]
        #duration_logits = torch.log_softmax(duration_logits, dim=-1)

        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.streaming_search(target_posteriors, targets, logits_lens, target_lens, t2skip)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist
    
    def streaming_search(
        self, target_posteriors: torch.Tensor, targets: torch.Tensor,  logits_lens: torch.tensor, target_lens: torch.Tensor, t2skip=None,
    ):
        assert t2skip is not None, 'User must provide "the skip infomation of each frame t", but found None.'
        B, T, U, _ = target_posteriors.shape
        assert B == 1, f"decoding batch size must be 1, get B = {B}"

        duration = "null"
        log_alpha = torch.zeros(B, T, U) - 1000 
        start_alpha = torch.zeros(B, T, U)
        total_alpha = torch.zeros(B, T, U)

        log_alpha = log_alpha.to(target_posteriors.device)

        # log field can't get 1. so use 1 to init log alpha.
        log_alpha_each_t = ["placeholder" for _ in range(T)]
        start_alpha_each_t = ["placeholder" for _ in range(T)]
        total_alpha_each_t = ["placeholder" for _ in range(T)]

        for b in range(B):
            t = 0
            while t < T:
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # this is the base case: (t=0, u=0) with log-alpha = 0.
                            log_alpha[b, t, u] = 0.0
                            start_alpha[b, t, u] = 0
                            total_alpha[b, t, u] = 1
                        else:
                            # this is case for (t = 0, u > 0), reached by (t, u - 1)
                            # emitting a blank symbol.
                            log_alpha[b, t, u] = 0.0
                            start_alpha[b, t, u] = t
                            total_alpha[b, t, u] = 1
                    else:
                        if t == 0:
                            # in case of (u > 0, t = 0), this is only reached from
                            # (t, u - 1) with a label emission.
                            log_alpha[b, t, u] = log_alpha[b, t, u - 1] + target_posteriors[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                            start_alpha[b, t, u] = t
                            total_alpha[b, t, u] = u + 1
                        else:
                            # here both t and u are > 0, this state is reachable
                            # with two possibilities: (t - duration, u) with a blank emissio or (t, u - 1) with a label emission.
                            from_left = log_alpha[b, t - duration, u] + target_posteriors[b, t - duration, u, self.blank]
                            from_down = log_alpha[b, t, u - 1] + target_posteriors[b, t, u - 1, targets[b, u - 1]]
                            if from_left >= from_down:
                                log_alpha[b, t, u] = from_left
                                start_alpha[b, t, u] = start_alpha[b, t-duration, u]
                                total_alpha[b, t, u] = total_alpha[b, t-duration, u] + 1
                            else:
                                log_alpha[b, t, u] = from_down
                                start_alpha[b, t, u] = start_alpha[b, t, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t, u-1] + 1
                                
                out_log_alpha = log_alpha[b, t, target_lens[b]] + target_posteriors[b, t, target_lens[b], self.blank]
                out_start_alpha = start_alpha[b, t, target_lens[b]]
                out_total_alpha = total_alpha[b, t, target_lens[b]] + 1 # the last blank transition.
                
                log_prob = out_log_alpha.item()
                log_alpha_each_t[t] = out_log_alpha
                start_alpha_each_t[t] = out_start_alpha.int().item()
                total_alpha_each_t[t] = out_total_alpha.int().item()
                
                duration = t2skip[t]
                t += duration
        
        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t