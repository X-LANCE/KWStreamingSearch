import torch

from KWStreamingSearch.base import KWSBaseSearch


class CTCFsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc_psd_decode = CTCPsdStreamingSearch(blank=blank, max_keep_blank_threshold=1.0)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        """The entry for CTC-based streaming search.
        
        Args:
            logits (torch.Tensor): B * T * V, B=1, V=len(vocabs + blank); 
                                   the logits output from the CTC-based model.
                                   Now we don't support batch decoding, so B=1.
            targets (torch.Tensor): B * U, B=1, U=len(keyword phoneme sequence); the target keyword sequence.
                                    Now we only support one arbitrary keyword search, so B=1.
            logits_lens (torch.tensor): B, B=1; the length of the logits. 
                                    logits_lens is not used for KWS-based search here. 
                                    But we don't remove it for compatibility. (ASR decoding requires it)
            target_lens (torch.Tensor): B, B=1; the length of the target keyword sequence (phoneme-based or subword-based).
        Returns:
            forward_logprob (float): forward_logprob is not used for KWS-based search here.
                                     But we don't remove it for compatibility. (ASR decoding uses it to observe the decoding confidence)
            logalpha_tlist (List[float]): the log probability of the keyword at each time step t.
            start_tlist (List[int]): start time step of the keyword at each time step t. 
                                     Used to observe the start time step of the keyword activation status.
            total_tlist (List[int]): total transition steps of the optimal keyword search path at each time step t.
                                     Used to normalize the activation score at each time step t.
                                     logalpha_tlist[t] / total_tlist[t] is almost the normalized activation score at time step t.
                                     (We use "almost" here as we may add a constant HIT_BONUS before nomalization.)
        """
        forward_logprob, logalpha_tlist, start_tlist, total_tlist \
            = self.ctc_psd_decode(logits, targets, logits_lens, target_lens)
        
        return forward_logprob, logalpha_tlist, start_tlist, total_tlist


class CTCPsdStreamingSearch(KWSBaseSearch):
    def __init__(self, blank: int = 0, max_keep_blank_threshold: float = 1.0):
        super().__init__(blank)
        assert max_keep_blank_threshold is not None and 0 < max_keep_blank_threshold <= 1.0
        # max_keep_blank_threshold should be less if PSD is more aggresive.
        self.prune_threshold = max_keep_blank_threshold
        self.blank = blank

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        posteriors = torch.log_softmax(logits, dim=-1)  # B T D

        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.streaming_search(posteriors, targets, logits_lens, target_lens)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist

    def streaming_search(
        self, posteriors: torch.Tensor, targets: torch.Tensor,  logits_lens: torch.tensor, target_lens: torch.Tensor
    ):
        B, T, _ = posteriors.shape
        U = max(target_lens).item()
        assert B == 1, f"decoding batch size must be 1, get B = {B}"
        assert isinstance(target_lens, torch.Tensor)
        # target = abc, vocab: u % 2 != 1, blank: u % 2 == 0, 
        # processed_target = [blank, a, blank, b, blank, c, blank]
        U_phi = 2 * U + 1       
        log_alpha = torch.zeros(B, T, U_phi) - 1000
        start_alpha = torch.zeros(B, T, U_phi)
        total_alpha = torch.zeros(B, T, U_phi)
        
        log_alpha = log_alpha.to(posteriors.device)

        log_prob = "placeholder"
        log_alpha_each_t = ["placeholder" for _ in range(T)] 
        start_alpha_each_t = ["placeholder" for _ in range(T)]
        total_alpha_each_t = ["placeholder" for _ in range(T)]

        b, t, psd_skips = 0, 0, 0 # B = 1
        while t < T:
            # for psd: if p_{t, `blank`} > self.prune_threshold, then skip current frame
            if torch.exp(posteriors[b, t, self.blank]) > self.prune_threshold:
                psd_skips += 1
                t += 1
                continue
            
            for u in range(U_phi):
                if u == 0 or u == 1:
                    # the initial entries for each time step (the first and second row.)
                    if t-psd_skips == 0:
                        # the first entry
                        log_alpha[b, t, u] = 0.0
                        start_alpha[b, t, u] = 0
                        total_alpha[b, t, u] = 1
                    else:
                        # other entries
                        log_alpha[b, t, u] = 0.0
                        start_alpha[b, t, u] = t 
                        total_alpha[b, t, u] = 1
                else:
                    # not initial entries
                    if t-psd_skips == 0:
                        # the first column
                        # not valid.
                        log_alpha[b, t, u] = -1e35
                        start_alpha[b, t, u] = -1
                        total_alpha[b, t, u] = 0
                    else:
                        # each time step
                        if u % 2 == 0:
                            # blank
                            from_blank = log_alpha[b, t-1-psd_skips, u] + posteriors[b, t-1-psd_skips, self.blank]
                            from_vocab = log_alpha[b, t-1-psd_skips, u-1] + posteriors[b, t-1-psd_skips, self.blank] # u or u-1?
                            
                            if from_blank >= from_vocab:
                                log_alpha[b, t, u] = from_blank
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u] + 1
                            else:
                                log_alpha[b, t, u] = from_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-1] + 1
                        else:
                            # vocab
                            from_last_vocab = log_alpha[b, t-1-psd_skips, u-2] + posteriors[b, t-1-psd_skips, targets[b, u//2]] # processed_target = [blank, a, blank, b, blank, c, blank]
                            from_blank = log_alpha[b, t-1-psd_skips, u-1] + posteriors[b, t-1-psd_skips, targets[b, u//2]]      # idx: u // 2 (\floor{u / 2})
                            from_now_vocab = log_alpha[b, t-1-psd_skips, u] + posteriors[b, t-1-psd_skips, targets[b, u//2]]
                            
                            max_score, max_index = torch.max(
                                torch.stack([from_last_vocab, from_blank, from_now_vocab], dim=0), dim=0)
                            
                            if max_index == 0:
                                # from last token
                                log_alpha[b, t, u] = from_last_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-2]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-2] + 1
                            elif max_index == 1:
                                # from blank
                                log_alpha[b, t, u] = from_blank 
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1-psd_skips, u-1] + 1
                            else:
                                # from current token
                                log_alpha[b, t, u] = from_now_vocab 
                                start_alpha[b, t, u] = start_alpha[b, t-1-psd_skips, u]
                                total_alpha[b, t, u] = total_alpha[0, t-1-psd_skips, u] + 1

            # output for each time step
            _, max_out_index = torch.max(
                torch.stack([log_alpha[b, t, U_phi-1], log_alpha[b, t, U_phi-2]], dim=0),dim=0)
            
            if max_out_index == 0:
                #  max path ends in a vocab token.
                out_row = U_phi - 1 
            else:
                # max path ends in a blk token.
                out_row = U_phi - 2
            
            out_log_alpha = log_alpha[b, t, out_row]
            out_start_alpha = start_alpha[b, t, out_row]
            out_total_alpha = total_alpha[b, t, out_row]

            log_prob = out_log_alpha.item()
            log_alpha_each_t[t] = out_log_alpha
            start_alpha_each_t[t] = out_start_alpha
            total_alpha_each_t[t] = out_total_alpha
            
            psd_skips = 0
            t += 1

        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t
