from typing import List

import torch
import torch.nn as nn

def logsumexp(a, b):
    ret = torch.logsumexp(torch.stack([a, b]), dim=0)
    return ret

def logsubexp(a, b):
    expa = torch.exp(a)
    expb = torch.exp(b)
    return torch.log(expa-expb)


class RNNTPrefixSearch(nn.Module):
    def __init__(self, blank_id: int, search_type='path'):
        super().__init__()
        self.blank = blank_id
        
        self.search_type = search_type
        
        # # pramas number 2/4
        # # /mnt/lustre/hpc_stor01/home/yu.xi_sx/workspace/hyl23/icassp2025/TDT-extension-07-15/extend_rnnts/model/rnnt.py:263
        # if search_type in ('path', 'max'):
        #     self.compute_forward_prob = self.kws_streaming_decoding
        # elif search_type == 'max_verbose':
        #     self.compute_forward_prob = self.kws_streaming_decoding_verbose
        # else:
        #     raise NotImplementedError

    def forward(self, logits, targets, logit_lens, target_lens):
        """
        Args:
            logits: (torch.Tensor) B x T x U x D, D = len(vocab_size) + 1(blank)
            targets: (torch.Tensor) B x U
            logit_lens: (torch.Tensor) B
            target_lens: (torch.Tensor) B
        """
        logits = torch.log_softmax(logits, dim=-1)
        
        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.kws_streaming_decoding_verbose(logits, targets, logit_lens, target_lens)
        
        return forward_logprob, logalpha_tlist, start_tlist, total_tlist

        # if self.search_type in ('path', 'max'):
        #     forward_logprob, logalpha_tlist = self.compute_forward_prob(logits, targets, logit_lens, target_lens)
        #     return forward_logprob, logalpha_tlist
        # elif self.search_type == 'max_verbose':
        #     forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
        #         self.compute_forward_prob(logits, targets, logit_lens, target_lens)
        #     return forward_logprob, logalpha_tlist, start_tlist, total_tlist
        # else:
        #     import ipdb; ipdb.set_trace()

    def kws_streaming_decoding(self, logits, targets, logit_lens, target_lens):
        B, T, U, _ = logits.shape

        Beam = 1
        log_alpha = torch.zeros(B, T, U, Beam) - 1000
        log_alpha[:, :, :, 0] = 0
        log_alpha = log_alpha.to(logits.device)
        
        log_alpha_each_t = []
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u, :] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - 1)
                        # emitting a blank symbol.
                        extra_path =  torch.zeros(B, dtype=torch.float32).to(logits.device)
                        log_alpha[:, t, u, 1:] = log_alpha[:, t - 1, u, :-1] + logits[:, t - 1, 0, self.blank].unsqueeze(-1)  # all the same, no need to sort
                        log_alpha[:, t, u, 0] = extra_path
                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            logits[:, t, u - 1, :], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1, 1)
                        log_alpha[:, t, u, :] = log_alpha[:, t, u - 1, :] + gathered.to(log_alpha.device)
                    else:
                        from_left = log_alpha[:, t - 1, u, :] + logits[:, t - 1, u, self.blank].unsqueeze(-1)
                        from_down = log_alpha[:, t, u - 1, :] + torch.gather(
                                        logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1, 1)    # index=targets[:, u - 1]: u - 1 because U = len(target + 1)
                        log_alpha[:, t, u, :] = torch.cat([from_left, from_down], dim=1).sort(dim=1, descending=True)[0][:, :Beam]

            log_probs = []
            for b in range(B):
                to_append = (
                    log_alpha[b, t, target_lens[b], 0] + logits[b, t, target_lens[b], self.blank]
                )
                log_probs.append(to_append)
            log_prob = torch.stack(log_probs)
            log_alpha_each_t.append(log_prob)
        '''
        # if you want to plot heatmap
        
        import numpy as np
        alpha_map = torch.exp(log_alpha[:,:,:,0]).squeeze(dim=0).cpu().numpy()  # T x U
        np.savetxt('alpha_map', alpha_map)  # save alpha map
        
        import matplotlib.pyplot as plt
        heatmap = plt.imshow(alpha_map.T, cmap='hot', interpolation='nearest')  # plot heatmap
        cbar = plt.colorbar(heatmap, label='Value')     # prepare legend
        plt.legend()                                    # place legend
        plt.gca().invert_yaxis()                        # reverse y-axix
        plt.savefig("alpha_map.png")                    # savefig :D
        '''
        return log_prob, log_alpha_each_t

    def kws_streaming_decoding_verbose(self, logits, targets, logit_lens, target_lens):
        B, T, U, _ = logits.shape
        assert B == 1, f"decoding batch size must be 1, get B = {B}"

        log_alpha = torch.zeros(B, T, U) - 1000
        # log_alpha[:, :, :] = 0
        log_alpha = log_alpha.to(logits.device)
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
                            logits[0, t, u - 1, :], dim=0, index=targets[0, u - 1].view(-1).type(torch.int64)).squeeze() # shape: [1] -> scalar
                        log_alpha[0, t, u] = log_alpha[0, t, u - 1] + gathered
                        start_alpha[0, t, u] = 0
                        total_alpha[0, t, u] = u + 1
                    else:
                        from_left = log_alpha[0, t - 1, u] + logits[0, t - 1, u, self.blank]

                        down_gathered = torch.gather(
                            logits[0, t, u - 1], dim=0, index=targets[0, u - 1].view(-1).type(torch.int64)).squeeze()
                        from_down = log_alpha[0, t, u - 1] + down_gathered

                        if from_left >= from_down:
                            log_alpha[0, t, u] = from_left
                            start_alpha[0, t, u] = start_alpha[0, t-1, u]
                            total_alpha[0, t, u] = total_alpha[0, t-1, u] + 1
                        else:
                            log_alpha[0, t, u] = from_down
                            start_alpha[0, t, u] = start_alpha[0, t, u-1]
                            total_alpha[0, t, u] = total_alpha[0, t, u-1] + 1

            out_log_alpha = log_alpha[0, t, target_lens[0]] + logits[0, t, target_lens[0], self.blank] # + reward
            out_start_alpha = start_alpha[0, t, target_lens[0]]
            out_total_alpha = total_alpha[0, t, target_lens[0]] + 1 # the last blank transition
            
            log_prob = out_log_alpha.item()
            log_alpha_each_t[t] = out_log_alpha
            start_alpha_each_t[t] = out_start_alpha.int().item()
            total_alpha_each_t[t] = out_total_alpha.int().item()

        # # if you want to plot heatmap
        # import numpy as np
        # alpha_map = torch.exp(log_alpha[:,:,:]).squeeze(dim=0).cpu().numpy()  # T x U
        # np.savetxt(f"heatmap/everything.rnnt.txt", alpha_map)  # save alpha map
        
        # import matplotlib.pyplot as plt
        # heatmap = plt.imshow(alpha_map.T, cmap='hot', interpolation='nearest')  # plot heatmap
        # plt.colorbar(heatmap, label='Value')     # prepare legend
        # plt.gca().invert_yaxis()                        # reverse y-axix
        # plt.legend()                                    # place legend
        # plt.savefig(f"rnnt-heatmaps/nihao_wenwen/alpha_map.{additional_conf['uid']}.png")                    # savefig :D
        # plt.clf()
        
        
        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t


class TDTPrefixSearch(nn.Module):
    def __init__(self, blank_id: int, durations: List[int], search_type='max'):
        super().__init__()
        self.blank = blank_id
        self.durations = durations
        self.n_durations = len(durations)
        self.search_type = search_type

        # if search_type == 'path' or search_type == 'max':
        #     self.compute_forward_prob = self.kws_streaming_decoding_with_gdy_duration_guided
        # elif search_type == 'max_verbose':
        #     self.compute_forward_prob = self.kws_streaming_decoding_with_gdy_duration_guided_verbose
        # else:
        #     raise NotImplementedError
    
    def logsumexp(self, a, b):
        ret = torch.logsumexp(torch.stack([a, b]), dim=0)
        return ret

    def forward(self, logits, targets, logit_lens, target_lens, t2skip):
        """
        Args:
            logits: (torch.Tensor) B x T x U x D, D = len(vocab_size) + 1(blank) + len(durations)
            targets: (torch.Tensor) B x U
            logit_lens: (torch.Tensor) B
            target_lensL (torch.Tensor) B
        """
        target_logits = logits[:, :, :, :-self.n_durations]
        duration_logits = logits[:, :, :, -self.n_durations:]

        target_logits = torch.log_softmax(target_logits, dim=-1)
        duration_logits = torch.log_softmax(duration_logits, dim=-1)

        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.kws_streaming_decoding_with_gdy_duration_guided_verbose(target_logits, duration_logits, targets, logit_lens, target_lens, t2skip)

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist

        # if self.search_type in ('path', 'max'): 
        #     forward_logprob, logalpha_tlist = self.compute_forward_prob(
        #         target_logits, duration_logits, targets, logit_lens, target_lens, t2skip)
        # elif self.search_type in ('max_verbose'):
        #     forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
        #         self.compute_forward_prob(target_logits, duration_logits, targets, logit_lens, target_lens, t2skip)

        # return forward_logprob, logalpha_tlist
    
    def kws_streaming_decoding_with_gdy_duration_guided_verbose(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
        assert t2skip is not None, 'User must provide "the skip infomation of each frame t", but found None.'
        B, T, U, _ = target_logits.shape
        assert B == 1, f"decoding batch size must be 1, get B = {B}"

        duration = "null"
        log_alpha = torch.zeros(B, T, U) - 1000 
        start_alpha = torch.zeros(B, T, U)
        total_alpha = torch.zeros(B, T, U)

        log_alpha = log_alpha.to(target_logits.device)

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
                            log_alpha[b, t, u] = log_alpha[b, t, u - 1] + target_logits[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                            start_alpha[b, t, u] = t
                            total_alpha[b, t, u] = u + 1
                        else:
                            # here both t and u are > 0, this state is reachable
                            # with two possibilities: (t - duration, u) with a blank emissio or (t, u - 1) with a label emission.
                            from_left = log_alpha[b, t - duration, u] + target_logits[b, t - duration, u, self.blank]
                            from_down = log_alpha[b, t, u - 1] + target_logits[b, t, u - 1, targets[b, u - 1]]
                            if from_left >= from_down:
                                log_alpha[b, t, u] = from_left
                                start_alpha[b, t, u] = start_alpha[b, t-duration, u]
                                total_alpha[b, t, u] = total_alpha[b, t-duration, u] + 1
                            else:
                                log_alpha[b, t, u] = from_down
                                start_alpha[b, t, u] = start_alpha[b, t, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t, u-1] + 1
                                
                out_log_alpha = log_alpha[b, t, target_lens[b]] + target_logits[b, t, target_lens[b], self.blank]
                out_start_alpha = start_alpha[b, t, target_lens[b]]
                out_total_alpha = total_alpha[b, t, target_lens[b]] + 1 # the last blank transition.
                
                log_prob = out_log_alpha.item()
                log_alpha_each_t[t] = out_log_alpha
                start_alpha_each_t[t] = out_start_alpha.int().item()
                total_alpha_each_t[t] = out_total_alpha.int().item()
                
                duration = t2skip[t]
                t += duration

        # if you want to plot heatmap
        # import numpy as np
        # alpha_map = torch.exp(log_alpha[:,:,:]).squeeze(dim=0).cpu().numpy()  # T x U
        # np.savetxt(f"heatmap/everything.tdt4.txt", alpha_map)  # save alpha map
        
        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t

    def kws_streaming_decoding_with_gdy_duration_guided(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
        assert t2skip is not None, 'User must provide "the skip infomation of each frame t", but found None.'
        
        B, T, U, _ = target_logits.shape

        duration = "null"
        Beam = 1
        log_alpha = torch.zeros(B, T, U, Beam) - 1000
        log_alpha = log_alpha.to(target_logits.device)

        # log field can't get 1. so use 1 to init log alpha.
        log_alpha_each_t = ["placeholder" for _ in range(T)] #log_alpha_each_t = [torch.ones(B, dtype=torch.float32, device=target_logits.device) for _ in range(T)]

        for b in range(B):
            t = 0
            while t < T:
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # this is the base case: (t=0, u=0) with log-alpha = 0.
                            log_alpha[b, t, u, :] = 0.0
                        else:
                            # this is case for (t = 0, u > 0), reached by (t, u - 1)
                            # emitting a blank symbol.
                            log_alpha[b, t, u, 1:] = log_alpha[b, t - duration, u, :-1] + target_logits[b, t - duration, 0, self.blank]  # all the same, no need to sort
                            log_alpha[b, t, u, 0] = 0.0
                    else:
                        if t == 0:
                            # in case of (u > 0, t = 0), this is only reached from
                            # (t, u - 1) with a label emission.
                            log_alpha[b, t, u, :] = log_alpha[b, t, u - 1, :] + target_logits[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1? Re: It's targets b, u-1!!!
                        else:
                            # here both t and u are > 0, this state is reachable
                            # with two possibilities: (t - duration, u) with a blank emissio or (t, u - 1) with a label emission.
                            from_left = log_alpha[b, t - duration, u, :] + target_logits[b, t - duration, u, self.blank]
                            from_down = log_alpha[b, t, u - 1, :] + target_logits[b, t, u - 1, targets[b, u - 1]]
                            log_alpha[b, t, u, :] = torch.cat([from_left, from_down], dim=0).sort(dim=0, descending=True)[0][:Beam]
               
                out_log_alpha = log_alpha[b, t, target_lens[b], 0] + target_logits[b, t, target_lens[b], self.blank]
                log_alpha_each_t[t] = out_log_alpha.item() # log_alpha_each_t[t][b] = out_log_alpha.item()
                log_prob = log_alpha_each_t[t]
                
                duration = t2skip[t]
                t += duration
        # import numpy as np
        # np.savetxt('belta_map', torch.exp(log_alpha[:,:,:,0]).squeeze(dim=0).cpu().numpy())
        return log_prob, log_alpha_each_t 


class CTCFsdPrefixSearch(nn.Module):
    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc_psd_decode = CTCPsdPrefixSearch(max_keep_blank_threshold=1.0, blank=blank) # max_keep_blank_threshold should be less if PDS is more aggresive.
        
    def forward(self, logits, targets, logit_lens, target_lens):
        forward_logprob, logalpha_tlist, start_tlist, total_tlist \
            = self.ctc_psd_decode(logits, targets, logit_lens, target_lens)
        
        return forward_logprob, logalpha_tlist, start_tlist, total_tlist


class CTCPsdPrefixSearch(nn.Module):
    def __init__(self, max_keep_blank_threshold: float, blank: int = 0):
        super().__init__()
        assert max_keep_blank_threshold is not None and 0 < max_keep_blank_threshold <= 1.0
        # assert blank == 0 and isinstance(blank, int), f'blank is invalid(For now, it should be), get blank = {blank}'
        # I don't know if here should be 0/71. 2024/7/12
    
        self.prune_threshold = max_keep_blank_threshold # if frame_blank_prob <= self.prune_threshold, keep the frame.
        self.blank = blank

    '''
    # 8230-279154-0041.trim.5-7
    ipdb> logits = logits[:, 25:, :]  # debug
    ipdb> torch.exp(torch.tensor(logalpha_tlist) / 8)[15]
    tensor(0.8832)
    ipdb> torch.exp(torch.tensor(logalpha_tlist1) / 11)[14]
    tensor(0.8894)
    '''

    def forward(self, logits, targets, logit_lens, target_lens):
        logits = torch.log_softmax(logits, dim=-1)  # B T D

        forward_logprob, logalpha_tlist, start_tlist, total_tlist = \
            self.kws_psd_ctc_streaming_decoding(logits, targets, logit_lens, target_lens)
        # prob = torch.exp(torch.tensor(logalpha_tlist))

        # forward_logprob1, logalpha_tlist1, start_tlist1, total_tlist1 = \
        #     self.kws_ctc_streaming_decoding(logits, targets, logit_lens, target_lens)
        # prob1 = torch.exp(torch.tensor(logalpha_tlist1))

        # import ipdb; ipdb.set_trace()

        return forward_logprob, logalpha_tlist, start_tlist, total_tlist

    def kws_ctc_streaming_decoding(self, logits, targets, logit_lens, target_lens):
        B, T, _ = logits.shape
        U = max(target_lens).item()
        assert B == 1, f"decoding batch size must be 1, get B = {B}"
        assert isinstance(target_lens, torch.Tensor)
        
        U_phi = 2 * U + 1
        log_alpha = torch.zeros(B, T, U_phi) - 1000 
        start_alpha = torch.zeros(B, T, U_phi)
        total_alpha = torch.zeros(B, T, U_phi)
        
        log_alpha = log_alpha.to(logits.device)

        log_alpha_each_t = [-114514 for _ in range(T)]
        start_alpha_each_t = ["placeholder" for _ in range(T)]
        total_alpha_each_t = ["placeholder" for _ in range(T)]
        
        b = 0 # B = 1
        for t in range(T):
            for u in range(U_phi):
                if u == 0 or u == 1:
                    # the initial entries for each time step (the first and second row.)
                    if t == 0:
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
                    if t == 0:
                        # the first column
                        # not valid.
                        log_alpha[b, t, u] = -1e35
                        start_alpha[b, t, u] = -1
                        total_alpha[b, t, u] = 0
                    else:
                        # each time step
                        if u % 2 == 0:
                            # blank
                            from_blank = log_alpha[b, t-1, u] + logits[b, t-1, self.blank]
                            from_vocab = log_alpha[b, t-1, u-1] + logits[b, t-1, self.blank] # u or u-1?
                            
                            if from_blank >= from_vocab:
                                log_alpha[b, t, u] = from_blank
                                start_alpha[b, t, u] = start_alpha[b, t-1, u]
                                total_alpha[b, t, u] = total_alpha[b, t-1, u] + 1
                            else:
                                log_alpha[b, t, u] = from_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1, u-1] + 1
                        else:
                            # vocab
                            from_last_vocab = log_alpha[b, t-1, u-2] + logits[b, t-1, targets[b, u//2]]
                            from_blank = log_alpha[b, t-1, u-1] + logits[b, t-1, targets[b, u//2]]
                            from_now_vocab = log_alpha[b, t-1, u] + logits[b, t-1, targets[b, u//2]]
                            
                            max_score, max_index = torch.max(
                                torch.stack([from_last_vocab, from_blank, from_now_vocab], dim=0), dim=0)
                            
                            if max_index == 0:
                                # from last token
                                log_alpha[b, t, u] = from_last_vocab
                                start_alpha[b, t, u] = start_alpha[b, t-1, u-2]
                                total_alpha[b, t, u] = total_alpha[b, t-1, u-2] + 1
                            elif max_index == 1:
                                # from blank
                                log_alpha[b, t, u] = from_blank 
                                start_alpha[b, t, u] = start_alpha[b, t-1, u-1]
                                total_alpha[b, t, u] = total_alpha[b, t-1, u-1] + 1
                            else:
                                # from current token
                                log_alpha[b, t, u] = from_now_vocab 
                                start_alpha[b, t, u] = start_alpha[b, t-1, u]
                                total_alpha[b, t, u] = total_alpha[0, t-1, u] + 1
            _, max_out_index = torch.max(
                torch.stack([log_alpha[b, t, U_phi-1], log_alpha[b, t, U_phi-2]], dim=0), dim=0)
            
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

        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t

    def kws_psd_ctc_streaming_decoding(self, logits, targets, logit_lens, target_lens):
        B, T, _ = logits.shape
        U = max(target_lens).item()
        assert B == 1, f"decoding batch size must be 1, get B = {B}"
        assert isinstance(target_lens, torch.Tensor)
                                # target = abc
                                # vocab: u % 2 != 1
                                # blank: u % 2 == 0
        U_phi = 2 * U + 1       # processed_target = [blank, a, blank, b, blank, c, blank]
        log_alpha = torch.zeros(B, T, U_phi) - 1000
        start_alpha = torch.zeros(B, T, U_phi)
        total_alpha = torch.zeros(B, T, U_phi)
        
        log_alpha = log_alpha.to(logits.device)

        log_prob = "placeholder"
        log_alpha_each_t = ["placeholder" for _ in range(T)] # log_alpha_each_t = ["placeholder" for _ in range(T)]
        start_alpha_each_t = ["placeholder" for _ in range(T)]
        total_alpha_each_t = ["placeholder" for _ in range(T)]

        # import pdb;pdb.set_trace();

        b, t, psd_skips = 0, 0, 0 # B = 1
        while t < T:
            # for psd: if p_{t, `blank`} > self.prune_threshold, then skip current frame
            if torch.exp(logits[b, t, self.blank]) > self.prune_threshold:
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
                            from_blank = log_alpha[b, t-1-psd_skips, u] + logits[b, t-1-psd_skips, self.blank]
                            from_vocab = log_alpha[b, t-1-psd_skips, u-1] + logits[b, t-1-psd_skips, self.blank] # u or u-1?
                            
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
                            from_last_vocab = log_alpha[b, t-1-psd_skips, u-2] + logits[b, t-1-psd_skips, targets[b, u//2]] # processed_target = [blank, a, blank, b, blank, c, blank]
                            from_blank = log_alpha[b, t-1-psd_skips, u-1] + logits[b, t-1-psd_skips, targets[b, u//2]]      # idx: u // 2 (\floor{u / 2})
                            from_now_vocab = log_alpha[b, t-1-psd_skips, u] + logits[b, t-1-psd_skips, targets[b, u//2]]
                            
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

        # import numpy as np
        # alpha_map = torch.exp(log_alpha[:,:,:]).squeeze(dim=0).cpu().numpy()  # T x U
        # np.savetxt("heatmap/everything.tdt4-ctc.txt", alpha_map)  # save alpha map
        # exit(-1)
        
        # import matplotlib.pyplot as plt
        # heatmap = plt.imshow(alpha_map.T, cmap='hot', interpolation='nearest')  # plot heatmap
        # cbar = plt.colorbar(heatmap, label='Value')     # prepare legend
        # plt.legend()                                    # place legend
        # plt.gca().invert_yaxis()                        # reverse y-axix
        # plt.savefig(f"heatmaps/ctc/alpha_map.{T}.png")                    # savefig :D
        # plt.clf()
        # from loguru import logger; logger.debug(f"frames: {T}")
        return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t















    # t-psd_skips，不是t-psd_skips-1
    # def kws_psd_ctc_streaming_decoding(self, logits, targets, logit_lens, target_lens):
    #     logits = logits[:, 25:, :]
    #     B, T, _ = logits.shape
    #     U = max(target_lens).item()
    #     assert B == 1, f"decoding batch size must be 1, get B = {B}"
    #     assert isinstance(target_lens, torch.Tensor)
    #                             # target = abc
    #                             # vocab: u % 2 != 1
    #                             # blank: u % 2 == 0
    #     U_phi = 2 * U + 1       # processed_target = [blank, a, blank, b, blank, c, blank]
    #     log_alpha = torch.zeros(B, T, U_phi) - 1e35
    #     start_alpha = torch.zeros(B, T, U_phi)
    #     total_alpha = torch.zeros(B, T, U_phi)
        
    #     log_alpha = log_alpha.to(logits.device)

    #     log_alpha_each_t = ["placeholder" for _ in range(T)]
    #     start_alpha_each_t = ["placeholder" for _ in range(T)]
    #     total_alpha_each_t = ["placeholder" for _ in range(T)]

    #     b, t, psd_skips = 0, 0, 0 # B = 1
    #     while t < T:
    #         # for psd: if p_{t, `blank`} > self.prune_threshold, then skip current frame
    #         if torch.exp(logits[b, t, self.blank]) > self.prune_threshold:
    #             psd_skips += 1
    #             t += 1
    #             continue
            
    #         for u in range(U_phi):
    #             if u == 0 or u == 1:
    #                 # the initial entries for each time step (the first and second row.)
    #                 if t-psd_skips == 0:
    #                     # the first entry
    #                     log_alpha[b, t, u] = 0.0
    #                     start_alpha[b, t, u] = 0
    #                     total_alpha[b, t, u] = 1
    #                 else:
    #                     # other entries
    #                     log_alpha[b, t, u] = 0.0
    #                     start_alpha[b, t, u] = t 
    #                     total_alpha[b, t, u] = 1
    #             else:
    #                 # not initial entries
    #                 if t-psd_skips == 0:
    #                     # the first column
    #                     # not valid.
    #                     log_alpha[b, t, u] = -1e35
    #                     start_alpha[b, t, u] = -1
    #                     total_alpha[b, t, u] = 0
    #                 else:
    #                     # each time step
    #                     if u % 2 == 0:
    #                         # blank
    #                         from_blank = log_alpha[b, t-psd_skips, u] + logits[b, t-psd_skips, self.blank]
    #                         from_vocab = log_alpha[b, t-psd_skips, u-1] + logits[b, t-psd_skips, self.blank] # u or u-1?
                            
    #                         if from_blank >= from_vocab:
    #                             log_alpha[b, t, u] = from_blank
    #                             start_alpha[b, t, u] = start_alpha[b, t-psd_skips, u]
    #                             total_alpha[b, t, u] = total_alpha[b, t-psd_skips, u] + 1
    #                         else:
    #                             log_alpha[b, t, u] = from_vocab
    #                             start_alpha[b, t, u] = start_alpha[b, t-psd_skips, u-1]
    #                             total_alpha[b, t, u] = total_alpha[b, t-psd_skips, u-1] + 1
    #                     else:
    #                         # vocab
    #                         from_last_vocab = log_alpha[b, t-psd_skips, u-2] + logits[b, t-psd_skips, targets[b, u//2]] # processed_target = [blank, a, blank, b, blank, c, blank]
    #                         from_blank = log_alpha[b, t-psd_skips, u-1] + logits[b, t-psd_skips, targets[b, u//2]]      # idx: u // 2 (\floor{u / 2})
    #                         from_now_vocab = log_alpha[b, t-psd_skips, u] + logits[b, t-psd_skips, targets[b, u//2]]
                            
    #                         max_score, max_index = torch.max(
    #                             torch.stack([from_last_vocab, from_blank, from_now_vocab], dim=0), dim=0)
                            
    #                         if max_index == 0:
    #                             # from last token
    #                             log_alpha[b, t, u] = from_last_vocab
    #                             start_alpha[b, t, u] = start_alpha[b, t-psd_skips, u-2]
    #                             total_alpha[b, t, u] = total_alpha[b, t-psd_skips, u-2] + 1
    #                         elif max_index == 1:
    #                             # from blank
    #                             log_alpha[b, t, u] = from_blank 
    #                             start_alpha[b, t, u] = start_alpha[b, t-psd_skips, u-1]
    #                             total_alpha[b, t, u] = total_alpha[b, t-psd_skips, u-1] + 1
    #                         else:
    #                             # from current token
    #                             log_alpha[b, t, u] = from_now_vocab 
    #                             start_alpha[b, t, u] = start_alpha[b, t-psd_skips, u]
    #                             total_alpha[b, t, u] = total_alpha[0, t-psd_skips, u] + 1

    #         # output for each time step
    #         _, max_out_index = torch.max(
    #             torch.stack([log_alpha[b, t, U_phi-1], log_alpha[b, t, U_phi-2]], dim=0),
    #             dim=0,
    #         )
            
    #         if max_out_index == 0:
    #             #  max path ends in a vocab token.
    #             out_row = U_phi - 1 
    #         else:
    #             # max path ends in a blk token.
    #             out_row = U_phi - 2
            
    #         out_log_alpha = log_alpha[b, t, out_row]
    #         out_start_alpha = start_alpha[b, t, out_row]
    #         out_total_alpha = total_alpha[b, t, out_row]

    #         log_prob = out_log_alpha.item()
    #         log_alpha_each_t[t] = out_log_alpha
    #         start_alpha_each_t[t] = out_start_alpha
    #         total_alpha_each_t[t] = out_total_alpha
            
    #         psd_skips = 0
    #         t += 1
    #     import ipdb; ipdb.set_trace()
    #     return log_prob, log_alpha_each_t, start_alpha_each_t, total_alpha_each_t











"""
    # rnnt, calculate prob of all paths.
    # no filler path.
    def compute_forward_prob_rnnt(self, logits, targets, logit_lens, target_lens):
        B, T, U, _ = logits.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(logits.device)

        log_alpha_each_t = []
        compute_step = 0
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - 1)
                        # emitting a blank symbol.
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + logits[:, t - 1, 0, self.blank]
                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from
                        # (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered.to(log_alpha.device)
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - 1, u) with a blank emission
                        # or (t, u - 1) with a label emission.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + logits[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )
            log_probs = []
            compute_step += 1
            for b in range(B):
                to_append = (
                    log_alpha[b, t, target_lens[b]] + logits[b, t, target_lens[b], self.blank]
                )
                log_probs.append(to_append)

            log_prob = torch.stack(log_probs)
            log_alpha_each_t.append(log_prob/compute_step)
        import numpy as np
        np.savetxt('alpha_map', log_alpha.squeeze(dim=0).cpu().numpy())
        return log_prob, log_alpha_each_t


    # rnnt, calculate prob of all paths.
    # introduce filler path.
    def compute_forward_prob_all_prob(self, logits, targets, logit_lens, target_lens):
        B, T, U, _ = logits.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(logits.device)

        log_alpha_each_t = []
        compute_step = 0
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - 1) emitting a blank symbol.
                        extra_path =  torch.zeros(B, dtype=torch.float32).to(logits.device)
                        log_alpha[:, t, u] = self.logsumexp(log_alpha[:, t - 1, u] + logits[:, t - 1, 0, self.blank], extra_path)
                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from(t, u - 1) with a label emission.
                        gathered = torch.gather(
                            logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered.to(log_alpha.device)
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - 1, u) with a blank emission
                        # or (t, u - 1) with a label emission.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + logits[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )
            log_probs = []
            compute_step += 1
            compute_step = 1
            for b in range(B):
                to_append = (
                    log_alpha[b, t, target_lens[b]] + logits[b, t, target_lens[b], self.blank]
                )
                log_probs.append(to_append)
            log_prob = torch.stack(log_probs)
            log_alpha_each_t.append(log_prob/compute_step)

        # import numpy as np
        # np.savetxt('alpha_map', log_alpha.squeeze(dim=0).cpu().numpy())
        return log_prob, log_alpha_each_t
"""

"""
    # calculate prob of all paths, guided by the greedy duration.
    def compute_naive_duration_guided_forward_prob(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
        assert t2skip is not None, f'User must provide "the skip infomation of each frame t", but found None.'
        
        B, T, U, _ = target_logits.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(target_logits.device)

        compute_step = 0
        # log field can't get 1. so use 1 to init log alpha.
        log_alpha_each_t = [torch.ones(B, dtype=torch.float32, device=target_logits.device) for _ in range(T)]
        for b in range(B):
            t = 0
            while t < T:
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # this is the base case: (t=0, u=0) with log-alpha = 0.
                            log_alpha[b, t, u] = 0.0
                        else:
                            # this is case for (t = 0, u > 0), reached by (t, u - 1)
                            # emitting a blank symbol.
                            log_alpha[b, t, u] = log_alpha[b, t - duration, u] + target_logits[b, t - duration, 0, self.blank]
                    else:
                        if t == 0:
                            # in case of (u > 0, t = 0), this is only reached from
                            # (t, u - 1) with a label emission.
                            log_alpha[b, t, u] = log_alpha[b, t, u - 1] + target_logits[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                        else:
                            # here both t and u are > 0, this state is reachable
                            # with two possibilities: (t - duration, u) with a blank emission
                            # or (t, u - 1) with a label emission.
                            log_alpha[b, t, u] = torch.logsumexp(
                                torch.stack(
                                    [
                                        log_alpha[b, t - duration, u] + target_logits[b, t - duration, u, self.blank],
                                        log_alpha[b, t, u - 1] + target_logits[b, t, u - 1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                                    ]
                                ),
                                dim=0,
                            )

                compute_step += 1
                out_log_alpha = log_alpha[b, t, target_lens[b]] + target_logits[b, t, target_lens[b], self.blank]
                log_alpha_each_t[t][b] = out_log_alpha.item() / compute_step
                log_prob = log_alpha_each_t[t]
                
                duration = t2skip[t]
                t += duration
        return log_prob, log_alpha_each_t


# calculate probs of all possible paths for tdt, following the way to cal tdt loss.
    def compute_topline_duration_guided_forwrad_prob(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
        B, T, U, _ = target_logits.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(target_logits.device)

        compute_step = 0
        # log field can't get 1. so use 1 to init log alpha.
        log_alpha_each_t = [torch.ones(B, dtype=torch.float32, device=target_logits.device) for _ in range(T)]
        for b in range(B):
            for t in range(T):
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # both t and u are 0, this is the base case for alphas.
                            log_alpha[b, t, u] = 0.0
                        else:
                            # u = 0 and t != 0: only considers blank emissions.
                            log_alpha[b, t, u] = -1000.0
                            for n, l in enumerate(self.durations):
                                if (
                                    t - l >= 0 and l > 0
                                ):  # checking conditions for blank emission, l has to be at least 1
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + target_logits[b, t - l, u, self.blank]
                                        + duration_logits[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                    else:
                        # u != 0 here, need to consider both blanks and non-blanks.
                        log_alpha[b, t, u] = -1000.0
                        for n, l in enumerate(self.durations):
                            if t - l >= 0:
                                if l > 0:  # for blank emissions. Need to ensure index is not out-of-bound.
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + target_logits[b, t - l, u, self.blank]
                                        + duration_logits[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                                # non-blank emissions.
                                tmp = (
                                    log_alpha[b, t - l, u - 1]
                                    + target_logits[b, t - l, u - 1, targets[b, u - 1]]
                                    + duration_logits[b, t - l, u - 1, n]
                                )
                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                compute_step += 1
                out_log_alpha = torch.Tensor([-1000]).to(target_logits.device)[0]
                # need to loop over all possible ways that blank with different durations contributes to the final forward alpha.
                for n, l in enumerate(self.durations):
                    if t - l >= 0 and l > 0:
                        bb = (
                            log_alpha[b, t, target_lens[b]] 
                            + target_logits[b, t - l, target_lens[b], self.blank]
                            + duration_logits[b, t - l, target_lens[b], n]
                        )
                        out_log_alpha = self.logsumexp(bb, 1.0 * out_log_alpha)
                log_alpha_each_t[t][b] = out_log_alpha.item() / compute_step
                log_prob = log_alpha_each_t[t]

        return log_prob, log_alpha_each_t

    # calculate prob of all paths.
    # introduce filler path.
    def compute_naive_duration_guided_forward_prob_allprob(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
        assert t2skip is not None, f'User must provide "the skip infomation of each frame t", but found None.'
        
        B, T, U, _ = target_logits.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(target_logits.device)

        compute_step = 0
        # log field can't get 1. so use 1 to init log alpha.
        log_alpha_each_t = [torch.ones(B, dtype=torch.float32, device=target_logits.device) for _ in range(T)]
        for b in range(B):
            t = 0
            while t < T:
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # this is the base case: (t=0, u=0) with log-alpha = 0.
                            log_alpha[b, t, u] = 0.0
                        else:
                            # this is case for (t = 0, u > 0), reached by (t, u - 1)
                            # emitting a blank symbol.
                            log_alpha[b, t, u] = self.logsumexp(log_alpha[b, t - duration, u] + target_logits[b, t - duration, 0, self.blank], torch.tensor(0, dtype=torch.float32).to(target_logits.device))
                    else:
                        if t == 0:
                            # in case of (u > 0, t = 0), this is only reached from
                            # (t, u - 1) with a label emission.
                            log_alpha[b, t, u] = log_alpha[b, t, u - 1] + target_logits[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                        else:
                            # here both t and u are > 0, this state is reachable
                            # with two possibilities: (t - duration, u) with a blank emission
                            # or (t, u - 1) with a label emission.
                            log_alpha[b, t, u] = torch.logsumexp(
                                torch.stack(
                                    [
                                        log_alpha[b, t - duration, u] + target_logits[b, t - duration, u, self.blank],
                                        log_alpha[b, t, u - 1] + target_logits[b, t, u - 1, targets[b, u - 1]] # targets b, u or targets b, u-1?
                                    ]
                                ),
                                dim=0,
                            )

                compute_step += 1
                compute_step = 1
                out_log_alpha = log_alpha[b, t, target_lens[b]] + target_logits[b, t, target_lens[b], self.blank]
                log_alpha_each_t[t][b] = out_log_alpha.item()
                log_prob = log_alpha_each_t[t]
                
                duration = t2skip[t]
                t += duration
        return log_prob, log_alpha_each_t 
"""

    # rnnt
    # def compute_forward_prob_of_alpha(self, logits, targets, logit_lens, target_lens):
    #     B, T, U, _ = logits.shape

    #     Beam = self.beam
    #     log_alpha = torch.zeros(B, T, U, Beam) - 1000
    #     log_alpha[:, :, :, 0] = 0
    #     log_alpha = log_alpha.to(logits.device)
        
    #     log_alpha_each_t = []
    #     for t in range(T):
    #         for u in range(U):
    #             if u == 0:
    #                 if t == 0:
    #                     # this is the base case: (t=0, u=0) with log-alpha = 0.
    #                     log_alpha[:, t, u, :] = 0.0
    #                 else:
    #                     # this is case for (t = 0, u > 0), reached by (t, u - 1)
    #                     # emitting a blank symbol.
    #                     extra_path =  torch.zeros(B, dtype=torch.float32).to(logits.device)
    #                     log_alpha[:, t, u, 1:] = log_alpha[:, t - 1, u, :-1] + logits[:, t - 1, 0, self.blank].unsqueeze(-1)  # all the same, no need to sort
    #                     log_alpha[:, t, u, 0] = extra_path
    #             else:
    #                 if t == 0:
    #                     # in case of (u > 0, t = 0), this is only reached from (t, u - 1) with a label emission.
    #                     gathered = torch.gather(
    #                         logits[:, t, u - 1, :], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
    #                     ).reshape(-1, 1)
    #                     log_alpha[:, t, u, :] = log_alpha[:, t, u - 1, :] + gathered.to(log_alpha.device)
    #                 else:
    #                     from_left = log_alpha[:, t - 1, u, :] + logits[:, t - 1, u, self.blank].unsqueeze(-1)
    #                     from_down = log_alpha[:, t, u - 1, :] + torch.gather(
    #                                     logits[:, t, u - 1], dim=1, index=targets[:, u - 1].view(-1, 1).type(torch.int64)
    #                                 ).reshape(-1, 1)
    #                     log_alpha[:, t, u, :] = torch.cat([from_left, from_down], dim=1).sort(dim=1, descending=True)[0][:, :Beam]

    #         log_probs = []
    #         for b in range(B):
    #             to_append = (
    #                 log_alpha[b, t, target_lens[b], 0] + logits[b, t, target_lens[b], self.blank]
    #             )
    #             log_probs.append(to_append)
    #         log_prob = torch.stack(log_probs)
    #         log_alpha_each_t.append(log_prob)

    #     # import numpy as np
    #     # np.savetxt('alpha_map', log_alpha.squeeze(dim=0).cpu().numpy())
    #     return log_prob, log_alpha_each_t

    # TDT
    # def compute_greedy_duration_guided_forward_prob_of_alpha(self, target_logits, duration_logits, targets, logit_lens, target_lens, t2skip=None):
    #     assert t2skip is not None, 'User must provide "the skip infomation of each frame t", but found None.'
        
    #     B, T, U, _ = target_logits.shape

    #     Beam = self.beam
    #     log_alpha = torch.zeros(B, T, U, Beam) - 1000
    #     log_alpha = log_alpha.to(target_logits.device)

    #     # log field can't get 1. so use 1 to init log alpha.
    #     log_alpha_each_t = [torch.ones(B, dtype=torch.float32, device=target_logits.device) for _ in range(T)]
    #     for b in range(B):
    #         t = 0
    #         while t < T:
    #             for u in range(U):
    #                 if u == 0:
    #                     if t == 0:
    #                         # this is the base case: (t=0, u=0) with log-alpha = 0.
    #                         log_alpha[b, t, u, :] = 0.0
    #                     else:
    #                         # this is case for (t = 0, u > 0), reached by (t, u - 1)
    #                         # emitting a blank symbol.
    #                         log_alpha[b, t, u, 1:] = log_alpha[b, t - duration, u, :-1] + target_logits[b, t - duration, 0, self.blank]  # all the same, no need to sort
    #                         log_alpha[b, t, u, 0] = 0.0
    #                 else:
    #                     if t == 0:
    #                         # in case of (u > 0, t = 0), this is only reached from
    #                         # (t, u - 1) with a label emission.
    #                         log_alpha[b, t, u, :] = log_alpha[b, t, u - 1, :] + target_logits[b, t, u-1, targets[b, u - 1]] # targets b, u or targets b, u-1?
    #                     else:
    #                         # here both t and u are > 0, this state is reachable
    #                         # with two possibilities: (t - duration, u) with a blank emissio or (t, u - 1) with a label emission.
    #                         from_left = log_alpha[b, t - duration, u, :] + target_logits[b, t - duration, u, self.blank]
    #                         from_down = log_alpha[b, t, u - 1, :] + target_logits[b, t, u - 1, targets[b, u - 1]]
    #                         log_alpha[b, t, u, :] = torch.cat([from_left, from_down], dim=0).sort(dim=0, descending=True)[0][:Beam]

    #             out_log_alpha = log_alpha[b, t, target_lens[b], 0] + target_logits[b, t, target_lens[b], self.blank]
    #             log_alpha_each_t[t][b] = out_log_alpha.item()
    #             log_prob = log_alpha_each_t[t]
                
    #             duration = t2skip[t]
    #             t += duration
    #     return log_prob, log_alpha_each_t 
