import math
import copy
import numpy as np
from typing import Optional, List
from attributedict.collections import AttributeDict

import torch
from loguru import logger
from .hypothesis import Hypothesis

from .prefix_search import RNNTPrefixSearch, TDTPrefixSearch, CTCFsdPrefixSearch, CTCPsdPrefixSearch
from ..predictor.utils import get_transducer_task_io
#from nemo.collections.asr.losses.rnnt import RNNTLoss

from collections import defaultdict

TIMEOUT = 100               # 30ms * TIMEOUT \approx length of audio in ms
# TIMEOUT = 70               # 30ms * TIMEOUT \approx length of audio in ms

class RNNTDecoder:
    def __init__(self,
                 model: torch.nn.Module,
                 predictor: torch.nn.Module,
                 jointer: torch.nn.Module,
                 blank_id: int,
                 search_type: str, 
                 beam_size: int,
                 keyword_ints: Optional[List[int]] = None,
                 max_symbols_per_step: Optional[int] = None,
                 durations: Optional[List] = None,
                 beamsearch_type: str = 'path',
                 score_norm: bool = True,
                 return_best_hypothesis: bool = True,
                 preserve_alignments: bool = False,
                 preserve_frame_confidence: bool = False,
                 ctc_blank_threshold: float = None,
                 is_ctc_alpha_norm: bool = True, 
                 is_transducer_alpha_norm: bool = False, 
                 joint_merge_cfg: AttributeDict = None,
                ):

        self.asr_model = model
        self.predictor = predictor
        self.jointer = jointer

        self.blank = blank_id
        self.sos = blank_id # start of sentence

        self.vocab_size = predictor.vocab_size # excluding blank.
        logger.warning('`self.blank` of RNNT/CTC Decoder: {}'.format(self.blank))

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size
        self.keyword_ints = keyword_ints
        self.score_norm = score_norm
        self.durations = durations

        self.search_type = search_type
        self.max_symbols = max_symbols_per_step
        self.return_best_hypothesis = return_best_hypothesis
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence

        self.ctc_blank_threshold = ctc_blank_threshold
        if self.ctc_blank_threshold is not None and search_type in ['ctc', 'joint']:
            logger.warning("ctc search type: {}SD with ctc_blank_threshold {}".format(
                'F' if self.ctc_blank_threshold == 1.0 else 'P', 
                ctc_blank_threshold,
            ))

        self.bonus = self._setup_bonus()

        self.is_transducer_alpha_norm = is_transducer_alpha_norm
        self.is_ctc_alpha_norm = is_ctc_alpha_norm
        self.joint_merge_cfg = joint_merge_cfg
        logger.warning('`is_ctc_alpha_norm` of {} is set `{}`. Be careful when wakeup word appears more than once. This works for ctc and joint.'.format(type(self).__name__, is_ctc_alpha_norm))

        if self.keyword_ints is None:
            # asr decode
            if self.beam_size == 1:
                if self.search_type == 'rnnt':
                    logger.warning("SEARCH TYPE: 'ASR Decode: rnnt greedy search' ")
                    self.search = self.rnnt_greedy_search 
                elif self.search_type == 'tdt':
                    logger.warning("SEARCH TYPE: 'ASR Decode: tdt greedy search.'")
                    self.search = self.tdt_greedy_search
                elif self.search_type == 'ctc':
                    logger.warning("SEARCH TYPE: 'ASR Decode: ctc greedy search for encoder & aux_decoder.'")
                    self.search = self.ctc_greedy_search
                else:
                    raise ValueError(f"invalid ASR Decode search_type: {self.search_type}")
            elif self.beam_size > 1:
                if self.search_type == 'rnnt':
                    logger.warning(f"SEARCH TYPE: 'ASR Decode: rnnt beam search', beam_size = {self.beam_size}")
                    self.search = self.rnnt_beam_search
                elif self.search_type == 'tdt':
                    logger.warning(f"SEARCH TYPE: 'ASR Decode: tdt beam search', beam_size = {self.beam_size}")
                    self.search = None
                    raise NotImplementedError('tdt beam search')
                elif self.search_type == 'ctc':
                    logger.warning(f"SEARCH TYPE: 'ASR Decode: ctc prefix beam search', beam_size = {self.beam_size}")
                    logger.warning("Note that for CTC, normal ASR beam search is meaningless, which is equal with ASR greedy search.")
                    self.search = self.ctc_prefix_beam_search
                else:
                    raise ValueError(f"invalid ASR Decode search_type: {self.search_type}")
        else:
            # kws decode, beam_size is useless
            if self.search_type == 'rnnt':
                logger.warning(f"SEARCH TYPE: 'KWS Decode: rnnt kws search.', search_type = {beamsearch_type}") # path(.) or alpha
                self.rnnt_prefix_search = RNNTPrefixSearch(blank_id=self.blank, search_type=beamsearch_type)
                self.search = self.rnnt_record_kwdscores_greedy_search 
            elif self.search_type == 'tdt':
                logger.warning(f"SEARCH TYPE: 'KWS Decode: tdt kws search.', search_type = {beamsearch_type}")
                assert isinstance(self.durations, List)
                self.tdt_prefix_search = TDTPrefixSearch(blank_id=self.blank, durations=self.durations, search_type=beamsearch_type)
                self.search = self.tdt_record_kwdscores_greedy_search
            elif self.search_type == 'ctc':
                logger.warning(f"SEARCH TYPE: 'KWS Decode: ctc kws search.', search_type = {beamsearch_type}")
                self.ctc_prefix_search = CTCPsdPrefixSearch(max_keep_blank_threshold=ctc_blank_threshold, blank=self.blank)
                self.search = self.ctc_record_kwdsscores_greedy_search
            elif self.search_type == 'joint':
                self.ctc_prefix_search = CTCPsdPrefixSearch(max_keep_blank_threshold=ctc_blank_threshold, blank=self.blank)
                self.search = self.joint_ctc_transducer_record_kwdsscores_greedy_search
                
                self.model_type = 'rnnt' if (self.durations is None or len(self.durations) == 0) else 'tdt'
                if self.model_type == 'rnnt':
                    self.rnnt_prefix_search = RNNTPrefixSearch(blank_id=self.blank, search_type=beamsearch_type)
                else:
                    assert isinstance(self.durations, List)
                    self.tdt_prefix_search = TDTPrefixSearch(blank_id=self.blank, durations=self.durations, search_type=beamsearch_type)
                logger.warning(f"SEARCH TYPE: 'KWS Decode: joint ctc-transudcer', search_params: {1}, model_type: {self.model_type}, joint merge config: {self.joint_merge_cfg}") # whether psd, whether tdt, how to merge scores. how to sync socres.

            else:
                raise ValueError(f"invalid KWS Decode search_type: {self.search_type}")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loop_times = 0
        self.inner_loop_times = 0
        self.predictor_times = 0
        self.jointer_times = 0
        self.steps = 0
        self.blank_log = []
        self.blank_counts = 0

    def _setup_bonus(self):
        if self.keyword_ints is None:
            return None
        assert len(self.keyword_ints) == 1
        l_kwd = len(self.keyword_ints[0])
        return {
            'ctc': 3.5, # l_kwd * 0 if self.keyword_ints[0] != [34, 30, 55, 45, 36, 53, 55] else 3.5,
            'transducer': 3.5, #l_kwd * 0 if self.keyword_ints[0] != [34, 30, 55, 45, 36, 53, 55] else 3.5,
        }

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses
        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.y_seq), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def rnnt_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        # return: hypothesis = [hyp1, hyp2, ...], meta = {}
        meta = {}
        meta['predicted_durations'] = []
        meta['predicted_ce_phonemes'] = []
        t = 0
        # encoder_out T, 1, D
        dec_state = self.predictor.initialize_state(encoder_out.transpose(0,1)) # the input of init_state is B x T x D

        hypothesis = Hypothesis(score=0.0, accum_score = [0.0] ,y_seq=[self.blank], dec_state=dec_state, timestep=[-1], last_token=None)
        #hypothesis = Hypothesis(score=0.0, y_seq=[self.blank], dec_state=state, timestep=[-1], length=encoder_len)
    
        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            hypothesis.alignments = [[]]
        
        if self.preserve_frame_confidence:
            hypothesis.frame_confidence = [[]]

        cache = {}
        # Init state and first token
        g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)

        import time
        start = time.perf_counter() 
        for time_idx in range(encoder_len):
            self.steps += 1
            f = encoder_out.narrow(dim=0, start=time_idx, length=1) # 1 x 1 x D

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0
           
            # While blank is not predicted, or we don't run out of max symbols per timestep
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                self.inner_loop_times += 1
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                jointer_start = time.perf_counter()
                ytu = torch.log_softmax(self.jointer(f.unsqueeze(dim=2), g.unsqueeze(dim=1)), dim=-1)
                self.jointer_times += (time.perf_counter() - jointer_start)
                ytu = ytu[0, 0, 0, :] # [V+1]
                
                # torch.max(0) op doesn't exist for FP 16.
                if ytu.dtype != torch.float32:
                    ytu = ytu.float()
                
                # get index k, of max prob
                logp, predk = torch.max(ytu, dim=-1)
                predk = predk.item()

                if self.preserve_alignments:
                    # insert logprobs into last timestep.
                    hypothesis.alignments[-1].append((ytu.to('cpu'), torch.tensor(predk, dtype=torch.int32)))

               # not implemented.
               # if self.preserve_frame_confidence:
               #     hypothesis.frame_confidence[-1].append(self._get_confidence(ytu))

                del ytu
                
                # If blank token is predicted, exit inner loop, move onto next timestep t.
                if predk == self.blank:
                    self.blank_log.append(logp)
                    self.blank_counts += 1
                    not_blank = False
                    if self.preserve_alignments:
                        hypothesis.alignments.append([]) # blank buffer for next timestep.
                    
                    if self.preserve_frame_confidence:
                        hypothesis.frame_confidence.append([]) # blank buffer for next ts.
                else:
                    # Update state and current sequence.
                    hypothesis.y_seq.append(int(predk))
                    hypothesis.accum_score.append(hypothesis.score + float(logp))
                    hypothesis.score += float(logp)
                    hypothesis.dec_state = state
                    hypothesis.timestep.append(time_idx)
                    hypothesis.last_token = int(predk)
                    # Compute next state and token
                    predictor_time = time.perf_counter()
                    g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)
                    predictor_dur = time.perf_counter() - predictor_time
                    self.predictor_times += predictor_dur
                
                # Increment token counter
                symbols_added += 1
                meta['predicted_durations'].append(1)
                meta['predicted_ce_phonemes'].append(int(predk))

        # Remove trailing empty list of alignments.
        if self.preserve_alignments:
            if len(hypothesis.alignments[-1]) == 0:
                del hypothesis.alignments[-1]
        
        if self.preserve_frame_confidence:
            if len(hypothesis.frame_confidence[-1]) == 0:
                del hypothesis.frmae_confidence[-1]
        
        # print(hypothesis.y_seq)
        end = time.perf_counter()
        self.loop_times += (end-start)
        #logger.error(f"total_times: {self.loop_times}")
        #logger.error(f"predictor_times: {self.predictor_times}")
        #logger.error(f"jointer_times: {self.jointer_times}")
        #logger.error(f"steps: {self.steps}")
        #logger.error(f"inner_steps: {self.inner_loop_times}")
        logger.error(f"blank_score: {math.exp(sum(self.blank_log)/self.blank_counts)}")
        return [hypothesis], meta

    def rnnt_beam_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        # encoder_out T, 1, D
        """Beam search implementation."""
        meta = {}
        # Initialize states.
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor([self.blank], device=encoder_out.device, dtype=torch.long)

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last toekn
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0
        
        # Initialize zero vector states
        dec_state = self.predictor.initialize_state(encoder_out.transpose(0,1)) # the input of init_state is B x T x D
        
        # Initialize first hypothesis for the beam (blank)
        kept_hyps = [Hypothesis(score=0.0, accum_score=[0.0], y_seq=[self.blank], dec_state=dec_state, timestep=[-1], length=0)]
        cache = {}
        
        if self.preserve_alignments:
            kept_hyps[0].alignments = [[]]
        
        for time_idx in range(int(encoder_len)):
            f = encoder_out.narrow(dim=0, start=time_idx, length=1) # 1 x 1 x D
            hyps = kept_hyps
            kept_hyps = []
            
            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)
                
                # update decoder state and get next score.
                g, state, lm_tokens = self.predictor.score_hypothesis(max_hyp, cache) # 1 x 1 x D
                
                # get next token
                ytu = torch.log_softmax(self.jointer(f.unsqueeze(dim=2), g.unsqueeze(dim=1)), dim=-1) # 1 x 1 x 1, V+1
                ytu = ytu[0, 0, 0, :] # [V+1]
                
                # preserve alignments
                if self.preserve_alignments:
                    logprobs = ytu.cpu().clone()
                
                # remove blank token before topk
                top_k = ytu[ids].topk(beam_k, dim=-1)

                # Two possible steps, blank token or non-blank token predicted
                ytu = (
                    torch.cat((top_k[0], ytu[self.blank].unsqueeze(dim=0))),
                    torch.cat((top_k[1]+index_incr, blank_tensor))
                )
                
                for logp, predk in zip(*ytu):
                    new_hyp = Hypothesis(
                        score=(max_hyp.score+float(logp)),
                        accum_score=max_hyp.accum_score[:],
                        y_seq=max_hyp.y_seq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        timestep=max_hyp.timestep[:],
                        length=encoder_len,
                    )
                    if self.preserve_alignments:
                        new_hyp.alignments = copy.deepcopy(max_hyp.alignments)

                    # if current token is blank, dont update sequence, just store the current hypothesis
                    if predk == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        # if non-blank token was predicted, update state and sequence and then search more hypothesis.
                        new_hyp.dec_state = state
                        new_hyp.y_seq.append(int(predk))
                        new_hyp.timestep.append(time_idx)
                        new_hyp.accum_score.append(max_hyp.score+float(logp))

                        hyps.append(new_hyp)
                    
                    if self.preserve_alignments:
                        if predk == self.blank:
                            new_hyp.alignments[-1].append(
                                (logprobs.clone(), torch.tensor(self.blank, dtype=torch.int32))
                            )
                        else:
                            new_hyp.alignments[-1].append(
                                (logprobs.clone(), torch.tensor(new_hyp.y_seq[-1], dtype=torch.int32))
                            )
                
                # keep those hypothesis that have scores greater than next search generation
                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score)

                # If enough hypothesis have scores gerater than next search generation, stop beam search.
                if len(kept_most_prob) >= beam:
                    if self.preserve_alignments:
                        for kept_h in kept_most_prob:
                            kept_h.alignments.append([]) # blank buffer for next timestep
                    
                    kept_hyps = kept_most_prob
                    break
        
        # Remvoe trailing empty list of alignments
        if self.preserve_alignments:
            for h in kept_hyps:
                if len(h.alignments[-1]) == 0:
                    del h.alignments[-1]
        
        sorted_beam_hyps = self.sort_nbest(kept_hyps)
        return sorted_beam_hyps, meta

    def tdt_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        # return: hypothesis = [hyp1, hyp2, ...], meta = {}
        meta = {}
        meta['predicted_durations'] = []
        meta['predicted_ce_phonemes'] = []
        meta['predicted_durations_entropy'] = []
        meta['predicted_durations_probs'] = []
        t = 0
        # encoder_out T, 1, D
        dec_state = self.predictor.initialize_state(encoder_out.transpose(0, 1)) # the input of init_state is B x T x D
        
        hypothesis = Hypothesis(score=0.0, accum_score=[0.0], y_seq=[self.blank], dec_state=dec_state, timestep=[-1], last_token=None)

        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            hypothesis.alignments = [[]]
        
        if self.preserve_frame_confidence:
            hypothesis.frame_confidence = [[]]
        
        cache = {}
        g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)
        
        import time
        start = time.perf_counter()
        time_idx = 0
        while time_idx < encoder_len:
            self.steps += 1
            # Extract encoder embedding at timestep t.
            # f = x[time_idx, :, :].unsqueeze(dim=0) # 1 x 1 x D
            f = encoder_out.narrow(dim=0, start=time_idx, length=1)
            
            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0
            
            need_loop = True
            while need_loop and (self.max_symbols is None or symbols_added < self.max_symbols):
                self.inner_loop_times += 1

                jointer_start = time.perf_counter()
                ytu_no_norm = self.jointer(f.unsqueeze(dim=2), g.unsqueeze(dim=1))
                self.jointer_times += (time.perf_counter() - jointer_start)

                ytu = torch.log_softmax(ytu_no_norm[0, 0, 0, :-len(self.durations)], dim=-1) # [V+1]
                d_logits = torch.log_softmax(ytu_no_norm[0, 0, 0, -len(self.durations):], dim=-1)

                # torch.max(0) op doesn't exist for FP 16.
                if ytu.dtype != torch.float32:
                    ytu = ytu.float()
                    d_logits = d_logits.float()

                ## append for calculating duration entropy
                d_probs = torch.softmax(d_logits, dim=-1)
                meta['predicted_durations_entropy'].append(-1 * torch.sum(d_probs * torch.log(d_probs), dim=-1).item())
                meta['predicted_durations_probs'].append(list(map(lambda x: f"{x:.4f}", d_probs.tolist())))
                ##

                # get index k, of max prob
                logp, predk = torch.max(ytu, dim=-1)
                predk = predk.item() # predk is the label at timestep t_s in inner loop, s >= 0
                
                d_logp, d_predk = torch.max(d_logits, dim=-1)
                d_predk = d_predk.item()
                
                skip = self.durations[d_predk]
                
                if self.preserve_alignments:
                    # insert logprobs into last timestep.
                    hypothesis.alignments[-1].append((ytu.to('cpu'), torch.tensor(predk, dtype=torch.int32)))

                # not implemented.
                # if self.preserve_frame_confidence:
                #     hypothesis.frame_confidence[-1].append(self._get_confidence(ytu)) 
                if skip != 0:
                    if additional_conf.get('preset_duration', -1) != -1:
                        skip = additional_conf['preset_duration']

                if 'preset_duration_list' in additional_conf.keys():
                    if len(additional_conf['preset_duration_list']) > t:
                        t_value = additional_conf['preset_duration_list'][t]
                        t += 1
                        skip = t_value

                del ytu

                # If blank token is predicted, exit inner loop, move onto next timestep t.
                if predk == self.blank:
                    not_blank = False
                    # this rarely happens, but we manually increment the `skip` number
                    # if blank is emitted and duration=0 is predicted. This prevents possible
                    # infinite loops.
                    if skip == 0:
                        skip = 1
                    
                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array.
                        hypothesis.alignments.append([]) # blank buffer for next timestep
                    
                    if self.preserve_frame_confidence:
                        hypothesis.frame_confidence.append([]) # blank buffer for next timestep
                else:
                    hypothesis.y_seq.append(int(predk))
                    hypothesis.accum_score.append(hypothesis.score + float(logp))
                    hypothesis.score += float(logp)
                    hypothesis.dec_state = state
                    hypothesis.timestep.append(time_idx)
                    hypothesis.last_token = int(predk)
                    # Compute next state and token
                    predictor_time = time.perf_counter()
                    g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)
                    predictor_dur = time.perf_counter() - predictor_time
                    self.predictor_times += predictor_dur
                
                # Increment token counter
                symbols_added += 1
                time_idx += skip
                need_loop = skip == 0
                meta['predicted_durations'].append(skip)
                meta['predicted_ce_phonemes'].append(int(predk))

            if self.max_symbols is not None:
                if symbols_added == self.max_symbols and skip == 0:
                    time_idx += 1
            
        # Remove trailing empty list of alignments.
        if self.preserve_alignments:
            if len(hypothesis.alignments[-1]) == 0:
                del hypothesis.alignments[-1]
        
        if self.preserve_frame_confidence:
            if len(hypothesis.frame_confidence[-1]) == 0:
                del hypothesis.frmae_confidence[-1]
                    
        # print(hypothesis.y_seq)
        end = time.perf_counter()
        self.loop_times += (end-start)
        
        # hyl23 edited 20230908 00:22 begin
        # comment where logger.error() are called
        # logger.error(f"total_times: {self.loop_times}")
        # logger.error(f"predictor_times: {self.predictor_times}")
        # logger.error(f"jointer_times: {self.jointer_times}")
        # logger.error(f"steps: {self.steps}")
        # logger.error(f"inner_steps: {self.inner_loop_times}")
        # hyl23 edited 20230908 00:22 end
        return [hypothesis], meta
    
    def tdt_beam_search(self, encoder_out: torch.Tensor, encoder_len: int):
        raise NotImplementedError
    
    def rnnt_record_kwdscores_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int): 
        meta = {}

        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels = torch.tensor(self.keyword_ints[0], device=self.device).unsqueeze(dim=0),
            encoder_out_lens = torch.tensor([encoder_len], device=self.device),
            ignore_id = self.blank,
            blank_id = self.blank,
        )
        
        decoder_out, x, y = self.predictor(decoder_in, u_len)

        joint_out = self.jointer(
            enc_out = encoder_out.transpose(0,1).unsqueeze(dim=2),
            dec_out = decoder_out.unsqueeze(dim=1)
        )

        # correct
        # prefix_loss, alpha = self.rnnt_prefix_search(joint_out, target, t_len, u_len)
        _, alpha, _, total_tlist = self.rnnt_prefix_search(joint_out, target, t_len, u_len)
        
        #logger.error("alpha: {}, total_tlist: {}".format(alpha, total_tlist))
        # 98.9
        # import ipdb; ipdb.set_trace()
        if self.is_transducer_alpha_norm:
            alpha = self.alpha_norm(alpha, total_tlist, 'transducer')
            #logger.error("normed alpha: {}".format(alpha))

        # import ipdb; ipdb.set_trace()
        # xiyu
        # import ipdb; ipdb.set_trace() <----->   logits = torch.log_softmax(logits[:, :, 0, 1:], dim=-1)  # B T D
        # prefix_loss, alpha, start_list, total_list = self.rnnt_prefix_search_fsd(joint_out, target, t_len, u_len)
        # prefix_loss2, alpha2, start_list2, total_list2 = self.rnnt_prefix_search_psd(joint_out, target, t_len, u_len) 
        return alpha, meta
        
    def tdt_record_kwdscores_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        meta = {}
        # set the index to get pre_set_duration_list
        t = 0
        meta['predicted_durations'] = []
        meta['predicted_durations_entropy'] = []
        meta['predicted_durations_probs'] = []

        # encoder_out T, 1, D
        dec_state = self.predictor.initialize_state(encoder_out.transpose(0, 1)) # the input of init_state is B x T x D
        
        hypothesis = Hypothesis(score=0.0, accum_score=[0.0], y_seq=[self.blank], dec_state=dec_state, timestep=[-1], last_token=None)

        cache = {}
        g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)

        t2skip = [0] * encoder_len
        t2maxscore = [1] * encoder_len
        t2label = [-1] * encoder_len
        time_idx = 0
        while time_idx < encoder_len:
            # Extract encoder embedding at timestep t.
            # f = x[time_idx, :, :].unsqueeze(dim=0) # 1 x 1 x D
            f = encoder_out.narrow(dim=0, start=time_idx, length=1)
            
            # Setup exit flags and counter
            symbols_added = 0
            
            need_loop = True
            while need_loop and (self.max_symbols is None or symbols_added < self.max_symbols):
                self.inner_loop_times += 1

                ytu_no_norm = self.jointer(f.unsqueeze(dim=2), g.unsqueeze(dim=1))

                ytu = torch.log_softmax(ytu_no_norm[0, 0, 0, :-len(self.durations)], dim=-1) # [V+1]
                d_logits = torch.log_softmax(ytu_no_norm[0, 0, 0, -len(self.durations):], dim=-1)

                ## append for calculating duration entropy
                d_probs = torch.softmax(d_logits, dim=-1)
                meta['predicted_durations_entropy'].append(-1 * torch.sum(d_probs * torch.log(d_probs), dim=-1).item())
                meta['predicted_durations_probs'].append(list(map(lambda x: f"{x:.4f}", d_probs.tolist())))
                ##

                # get index k, of max prob
                logp, predk = torch.max(ytu, dim=-1)
                predk = predk.item() # predk is the label at timestep t_s in inner loop, s >= 0
                
                d_logp, d_predk = torch.max(d_logits, dim=-1)
                d_predk = d_predk.item()
                
                skip = self.durations[d_predk]
                # decoding
                '''
                # priority: for model -> for skip
                # ft_data: pinf, 0, -5, -10, -15
                # deode_snr: pinf, 0, -5, -10, -15
                # recall <--relation--> duration
                if skip != 0:
                    logger.warning(f"Set skip to fixed value `{skip}`.")
                    skip = 1 # 1, 2, 3, 4
                '''
                if skip != 0:
                    if additional_conf.get('preset_duration', -1) != -1:
                        skip = additional_conf['preset_duration']

                if 'preset_duration_list' in additional_conf.keys():
                    if len(additional_conf['preset_duration_list']) > t:
                        t_value = additional_conf['preset_duration_list'][t]
                        t += 1
                        skip = t_value

                del ytu

                t2label[time_idx] = predk
                # If blank token is predicted, exit inner loop, move onto next timestep t.
                if predk == self.blank:
                    # this rarely happens, but we manually increment the `skip` number
                    # if blank is emitted and duration=0 is predicted. This prevents possible
                    # infinite loops.
                    if skip == 0:
                        skip = 1
                else:
                    hypothesis.y_seq.append(int(predk))
                    hypothesis.accum_score.append(hypothesis.score + float(logp))
                    hypothesis.score += float(logp)
                    hypothesis.dec_state = state
                    hypothesis.timestep.append(time_idx)
                    hypothesis.last_token = int(predk)
                    # Compute next state and token
                    g, state, _ = self.predictor.score_hypothesis(hypothesis, cache)
                
                # Increment token counter
                t2skip[time_idx] += skip
                t2maxscore[time_idx] = hypothesis.score 

                symbols_added += 1
                time_idx += skip
                need_loop = skip == 0
                meta['predicted_durations'].append(skip)

                    
            if self.max_symbols is not None:
                if symbols_added == self.max_symbols and skip == 0:
                    t2skip[time_idx] += 1
                    time_idx += 1

        # ----^ get duration list from pre_train.^ ----
        # ---- test. ----
        # print(meta['predicted_durations'])

        # ----^ get duration info from greedy path.^ ----
        # ---- test. ----
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels = torch.tensor(self.keyword_ints[0], device=self.device).unsqueeze(dim=0),
            encoder_out_lens = torch.tensor([encoder_len], device=self.device),
            ignore_id = self.blank,
            blank_id = self.blank,
        )
        
        #  torch.tensor([len(self.keyword_ints[0])])
        decoder_out, x, y = self.predictor(decoder_in, u_len)

        joint_out = self.jointer(
            enc_out = encoder_out.transpose(0,1).unsqueeze(dim=2),
            dec_out = decoder_out.unsqueeze(dim=1)
        )
        
        # prefix_loss, alpha = self.tdt_prefix_search(joint_out, target, t_len, u_len, t2skip)
        _, alpha, _, total_tlist = self.tdt_prefix_search(joint_out, target, t_len, u_len, t2skip)

        if self.is_transducer_alpha_norm:
            alpha = self.alpha_norm(alpha, total_tlist, 'transducer')

        return alpha, meta

### following codes are for ctc decoding ###

    def ctc_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        assert additional_conf['aux_decode']['loss_name'] == 'ctc'
        assert encoder_out.shape[1] == 1, "Batch size needs to be 1, but gets `{}`.".format(encoder_out.shape[1])
        # assert encoder_out.shape[0] == 1, "Batch size needs to be 1, but gets `{}`.".format(encoder_out.shape[0])

        # warning: asr decoding s # e # e 
        # the second e cannot directly transfer to next e
        # bash decode.sh --search_type ctc --exp exp/aux-ctc/librispeech/rnnt_aux_0.3ctc --keywords everything --subsets test_clean --datasets LS20 --beam_size 1 --decode_algorithm asr

        meta = {}
        hypothesis = Hypothesis(
            score=0.0,
            accum_score=[0.0],
            y_seq=[self.blank],
            timestep=[-1], 
        )

        aux_decoder_out = additional_conf['aux_decode']['aux_decoder_out']
        logits = torch.log_softmax(aux_decoder_out, dim=-1)  # B T D, B should be 1 and only 1

        b = 0

        for time_idx in range(encoder_len):
            ytu = logits[b, time_idx, :]    # 1 x 72

            logp, predk = torch.max(ytu, dim=-1)
            predk = predk.item()

            del ytu

            if predk == self.blank:
                self.blank_log.append(logp)
                self.blank_counts += 1
            else:
                hypothesis.y_seq.append(int(predk))
                hypothesis.accum_score.append(hypothesis.score + float(logp))
                hypothesis.score += float(logp)
                hypothesis.timestep.append(time_idx)
                hypothesis.last_token = int(predk)

        logger.error(f"blank_score: {math.exp(sum(self.blank_log)/self.blank_counts)}")
        return [hypothesis], meta

    def ctc_prefix_beam_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        assert additional_conf['aux_decode']['loss_name'] == 'ctc'
        # encoder_out T, 1, D
        """Beam search implementation."""

        meta = {}

        aux_decoder_out = additional_conf['aux_decode']['aux_decoder_out']  # B T D
        B, T, V = aux_decoder_out.shape

        logits = torch.log_softmax(aux_decoder_out, dim=-1)
        meta['blank_score'] =  torch.exp(logits[0, :, self.blank])

        # Initialize states.
        beam = min(self.beam_size, self.vocab_size) # for len(kept_beams), 
        beam_k = min(beam, (self.vocab_size - 1))   # for expension, score_beam_size
        prune_threshold = 0.05

        # Initialize first hypothesis for the beam (blank)
        # cur_hyps <--> kept_hyps
        # from wekws
        # 2. CTC beam search step by step
        logits = logits.squeeze()
        ctc_probs = logits.exp()
        

        cur_hyps = [(tuple(), (1.0, 0.0, []))]

        # 2. CTC beam search step by step
        for t in range(0, T):
            probs = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (0.0, 0.0, []))

            # 2.1 First beam prune: select topk best
            top_k_probs, top_k_index = probs.topk(beam_k)  # (score_beam_size,)

            # filter prob score that is too small
            filter_probs = []
            filter_index = []
            for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
                if prob > prune_threshold:
                    filter_probs.append(prob)
                    filter_index.append(idx)

            if len(filter_index) == 0:
                continue

            for s in filter_index:
                ps = probs[s].item()
                for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == self.blank:  # blank
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pb = n_pb + pb * ps + pnb * ps
                        nodes = cur_nodes.copy()
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)
                    elif s == last:
                        if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                            # Update *ss -> *s;
                            n_pb, n_pnb, nodes = next_hyps[prefix]
                            n_pnb = n_pnb + pnb * ps
                            nodes = cur_nodes.copy()
                            if ps > nodes[-1]['prob']:  # update frame and prob
                                nodes[-1]['prob'] = ps
                                nodes[-1]['frame'] = t
                            next_hyps[prefix] = (n_pb, n_pnb, nodes)

                        if not math.isclose(pb, 0.0, abs_tol=0.000001):
                            # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb, nodes = next_hyps[n_prefix]
                            n_pnb = n_pnb + pb * ps
                            nodes = cur_nodes.copy()
                            nodes.append(dict(token=s, frame=t,
                                            prob=ps))  # to record token prob
                            next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        if nodes:
                            if ps > nodes[-1]['prob']:  # update frame and prob
                                # nodes[-1]['prob'] = ps
                                # nodes[-1]['frame'] = t
                                # avoid change other beam which has this node.
                                nodes.pop()
                                nodes.append(dict(token=s, frame=t, prob=ps))
                        else:
                            nodes = cur_nodes.copy()
                            nodes.append(dict(token=s, frame=t,
                                            prob=ps))  # to record token prob
                        n_pnb = n_pnb + pb * ps + pnb * ps
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

            cur_hyps = next_hyps[:beam]

        hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
        kept_hyps = []
        for hyp in hyps:
            score_list = [math.log(_['prob']) for _ in hyp[-1]]
            accum_score_list = [0]
            for score_idx in range(len(score_list)):
                accum_score_list.append(accum_score_list[-1] + score_list[score_idx])

            processed_hyp = Hypothesis(
                y_seq=[self.blank] + list(hyp[0]),
                score=sum(score_list),
                accum_score=[0.0] + accum_score_list,
                timestep=[-1] + [_['frame'] for _ in hyp[-1]],
            )

            kept_hyps.append(processed_hyp)

        if len(kept_hyps) == 0:
            kept_hyps = [Hypothesis(
                score=0.0,
                accum_score=[0.0],
                y_seq=[self.blank],
                timestep=[-1], 
            )]

        sorted_beam_hyps = kept_hyps
        return sorted_beam_hyps, meta

    def ctc_record_kwdsscores_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        '''
            additional_conf:
                'loss_name': ctc
        '''
        assert additional_conf['aux_decode']['loss_name'] == 'ctc'

        meta = {}
        # meta['predicted_durations'] = []
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels = torch.tensor(self.keyword_ints[0], device=self.device).unsqueeze(dim=0),
            encoder_out_lens = torch.tensor([encoder_len], device=self.device),
            ignore_id = self.blank,
            blank_id = self.blank,
        )

        aux_decoder_out = additional_conf['aux_decode']['aux_decoder_out']  # B T D
        
        logits = torch.log_softmax(aux_decoder_out, dim=-1)
        meta['blank_score'] =  torch.exp(logits[0, :, self.blank])
        # import ipdb; ipdb.set_trace()
        _, alpha, start_tlist, total_tlist = self.ctc_prefix_search(aux_decoder_out, target, t_len, u_len)

        if self.is_ctc_alpha_norm:
            # warning, consider multi keywords in 1 utterance
            # margin，找极值点
            # wkp_ind = torch.argmax(torch.tensor(alpha))
            # wkp_path_length = total_tlist[wkp_ind]
            # alpha = [_ / wkp_path_length for _ in alpha]
            # import ipdb; ipdb.set_trace()
            alpha = self.alpha_norm(alpha, total_tlist)

        return alpha, meta
    
    # def alpha_norm(self, alpha, total_tlist, joint_flag=None, logits=None, start_tlist=None):
    def alpha_norm(self, alpha, total_tlist, joint_flag=None):
        hit_bonus = 0
        if self.search_type == 'ctc':
            hit_bonus = self.bonus['ctc']
        elif self.search_type in ('rnnt', 'tdt'):
            hit_bonus = self.bonus['transducer']
        else:   #joint
            assert joint_flag in ('transducer', 'ctc')
            if joint_flag == 'ctc':
                hit_bonus = self.bonus['ctc']
            else:
                hit_bonus = self.bonus['transducer']

        ## reward
        logger.warning('HIT_BONUS for norm: {}'.format(hit_bonus))
        alpha = [_ + hit_bonus if _ != 'placeholder' else 'placeholder' for _ in alpha]

        ## timeout
        logger.warning('TIMEOUT for norm: {}'.format(TIMEOUT))
        for t in range(len(alpha)):
            if alpha[t] == 'placeholder':
                continue

            # if total_tlist[t] > TIMEOUT:
            if total_tlist[t] > TIMEOUT:
                alpha[t] = -1e35

        # ## minus_blank
        # logger.warning('Minus blank scores')
        # for t in range(len(alpha)):
        #     ts = int(start_tlist[t])
        #     t_dur = int(total_tlist[t])
        #     if ts < 0:
        #         continue
        #     alpha[t] -= logits[:, ts: ts + t_dur, self.blank].sum()

        # import ipdb; ipdb.set_trace()
        normed_alpha = [(alpha[t] / total_tlist[t] if total_tlist[t] != 'placeholder' else 'placeholder') for t in range(len(alpha))]
        normed_alpha = [torch.Tensor([_]).to(self.device) if isinstance(_, float) else _ for _ in normed_alpha]
        # logger.debug(f"alpha: {alpha}")
        # logger.debug(f"path lengths at each time step: {total_tlist}")
        # logger.debug(f"normed_alpha: {normed_alpha}")
        # logger.debug(f"norm_path_length: {total_tlist[torch.argmax(torch.tensor(alpha)).item()]}")
        # logger.debug(f"frame idx of maximum score (unnormed): {torch.argmax(torch.tensor(alpha))}")
        # logger.debug(f"frame idx of maximum score (normed)  : {torch.argmax(torch.tensor(normed_alpha))}")
        return normed_alpha

    def joint_ctc_transducer_record_kwdsscores_greedy_search(self, encoder_out: torch.Tensor, encoder_len: int, additional_conf: dict = {}):
        assert additional_conf['aux_decode']['loss_name'] == 'ctc'
        assert self.model_type in ('rnnt', 'tdt')

        logger.warning('`is_transducer_alpha_norm` of {} is set `{}`'.format(type(self).__name__, self.is_transducer_alpha_norm))


        meta = {}

        _, target, t_len, u_len = get_transducer_task_io(
            labels = torch.tensor(self.keyword_ints[0], device=self.device).unsqueeze(dim=0),
            encoder_out_lens = torch.tensor([encoder_len], device=self.device),
            ignore_id = self.blank,
            blank_id = self.blank,
        )

        # assert self.model_type == 'rnnt', 'TDT joint search has not been implemented yet.'

        # calculate ctc decoding
        aux_decoder_out = additional_conf['aux_decode']['aux_decoder_out']  # B T D

        _, ctc_alpha, _, ctc_total_tlist = self.ctc_prefix_search(aux_decoder_out, target, t_len, u_len)
        if self.is_ctc_alpha_norm:
            ctc_alpha = self.alpha_norm(ctc_alpha, ctc_total_tlist, joint_flag='ctc')

        # ctc_wkp_ind = torch.argmax(torch.tensor(ctc_alpha))
        # ctc_wkp_path_length = ctc_total_tlist[ctc_wkp_ind - 5: ctc_wkp_ind + 5]

        # calculate transducer decoding
        if self.model_type == 'rnnt':
            transducer_alpha, _ = self.rnnt_record_kwdscores_greedy_search(encoder_out, encoder_len)
        else:
            transducer_alpha, _ = self.tdt_record_kwdscores_greedy_search(encoder_out, encoder_len, additional_conf)
        # rnnt_wkp_ind = torch.argmax(torch.tensor(transducer_alpha))
        # rnnt_wkp_path_length = transducer_total_tlist[rnnt_wkp_ind - 5: rnnt_wkp_ind + 5]


        assert len(transducer_alpha) == len(ctc_alpha), 'The length of `transducer_alpha` and `ctc_alpha` are not equal: {}, {}.'.format(len(transducer_alpha), len(ctc_alpha))

        alpha = self._join_transducer_and_ctc_scores(transducer_alpha, ctc_alpha)

        # keyword="nihao_wenwen"
        # np.savetxt(f"{additional_conf['uid']}.{keyword}.ctc", torch.tensor(ctc_alpha).cpu().numpy())
        # np.savetxt(f"{additional_conf['uid']}.{keyword}.transducer", torch.tensor(transducer_alpha).cpu().numpy())
        # np.savetxt(f"{additional_conf['uid']}.{keyword}.joint", torch.tensor(alpha).cpu().numpy())
        # import ipdb; ipdb.set_trace()
        # np.savetxt("heatmap/everything.tdt4-joint.txt", torch.tensor(alpha).cpu().numpy())
        return alpha, meta

    # 注意以后做分数融合不要在log域做加法，而是要先exp再加，最后如果有必要再换回ln (torch.log)
    def _join_transducer_and_ctc_scores(self, transducer_alpha, ctc_alpha):
        alpha = []
        if self.joint_merge_cfg.async_guider == 'vector_wise':
            alpha = self.vector_wise_score_merge(transducer_alpha, ctc_alpha)
            
            return alpha
        for _ in range(len(transducer_alpha)):
            if self.joint_merge_cfg.async_guider == 'either':
                if transducer_alpha[_] == 'placeholder' and ctc_alpha[_] == 'placeholder':
                    alpha.append('placeholder')
                elif transducer_alpha[_] != 'placeholder' and ctc_alpha[_] != 'placeholder':
                    merged_score = self._one_step_score_merge(transducer_alpha[_], ctc_alpha[_])
                    alpha.append(merged_score)
                elif transducer_alpha[_] == 'placeholder':
                    alpha.append(ctc_alpha[_])
                else:
                    alpha.append(transducer_alpha[_])
            elif self.joint_merge_cfg.async_guider == 'transducer':
                if transducer_alpha[_] != 'placeholder':
                    alpha.append(transducer_alpha[_])
                else:
                    alpha.append(ctc_alpha[_])
            elif self.joint_merge_cfg.async_guider == 'ctc':
                if ctc_alpha[_] != 'placeholder':
                    alpha.append(ctc_alpha[_])
                else:
                    alpha.append(transducer_alpha[_])
            else:
                import pdb; pdb.set_trace()
                raise NotImplementedError
    
        return alpha

    # calculate the cosine_similarity of list[vec1] and list[vec2]
    def cosine_similarity(self, vec1, vec2):
        vec1 = [torch.tensor(-1.0000e+35) if x == 'placeholder' else x for x in vec1]
        vec2 = [torch.tensor(-1.0000e+35) if x == 'placeholder' else x for x in vec2]

        dot_product = sum(torch.exp(a) * torch.exp(b) for a, b in zip(vec1, vec2))
        norm_vec1 = math.sqrt(sum(torch.exp(a) * torch.exp(a) for a in vec1))
        norm_vec2 = math.sqrt(sum(torch.exp(b) * torch.exp(b) for b in vec2))
        
        return dot_product / (norm_vec1 * norm_vec2 + 1e-30)
    
    def cosine_similarity_remove(self, vec1, vec2):
        vec1_continuous = []
        vec2_continuous = []
        for i in range(len(vec1)):
            if vec1[i] != 'placeholder' and vec2[i] != 'placeholder':
                vec1_continuous.append(vec1[i])
                vec2_continuous.append(vec2[i])
        vec1 = vec1_continuous
        vec2 = vec2_continuous
        dot_product = sum(torch.exp(a) * torch.exp(b) for a, b in zip(vec1, vec2))
        norm_vec1 = math.sqrt(sum(torch.exp(a) * torch.exp(a) for a in vec1))
        norm_vec2 = math.sqrt(sum(torch.exp(b) * torch.exp(b) for b in vec2))
        
        return dot_product / (norm_vec1 * norm_vec2 + 1e-30)
    
    def padding_continuous(self, vec1, vec2):
        vec1_pre = torch.tensor(-1.0000e+35)
        vec2_pre = torch.tensor(-1.0000e+35)
        for i in range(len(vec1)):
            if vec1[i] != 'placeholder':
                vec1_pre = vec1[i]
            if vec2[i] != 'placeholder':
                vec2_pre = vec2[i]
            if vec1[i] == 'placeholder':
                vec1[i] = vec1_pre
            if vec2[i] == 'placeholder':
                vec2[i] = vec2_pre
        return vec1,vec2

    
    # vector-wise merge
    def vector_wise_score_merge(self, transducer_score, ctc_score):
        if self.joint_merge_cfg.joint_method.operation == 'cos_similarity':
            alpha = []
            window_size = self.joint_merge_cfg.joint_method.window_size
            if self.joint_merge_cfg.joint_method.standards == 'right':
                if (len(transducer_score) - window_size + 1) > 0:
                    for i in range(len(transducer_score) - window_size + 1): 
                        window_ctc = ctc_score[i:i+window_size]
                        window_trans = transducer_score[i:i+window_size]
                        similarity = self.cosine_similarity(window_ctc, window_trans)
                        weight = max(0, similarity)  
                        fused_score = (torch.exp(transducer_score[i]) + weight * torch.exp(ctc_score[i])) / (1 + weight)
                        alpha.append(fused_score)
                    return alpha
                else:
                    for i in range(len(transducer_score)):
                        merged_score = torch.log(0.5 * torch.exp(transducer_score[i]) + 0.5 * torch.exp(ctc_score[i]))
                        alpha.append(merged_score)
                    return alpha
            
            elif self.joint_merge_cfg.joint_method.standards == 'left':
                if (len(transducer_score) - window_size + 1) > 0:
                    for i in range(len(transducer_score)): 
                        left = max(0, i-window_size+1)
                        window_ctc = ctc_score[left:i]
                        window_trans = transducer_score[left:i]
                        similarity = self.cosine_similarity(window_ctc, window_trans)
                        weight = max(0, similarity) 
                        fused_score = (torch.exp(transducer_score[i]) + weight * torch.exp(ctc_score[i])) / (1 + weight)
                        alpha.append(fused_score)
                    return alpha
                else:
                    for i in range(len(transducer_score)):
                        merged_score = torch.log(0.5 * torch.exp(transducer_score[i]) + 0.5 * torch.exp(ctc_score[i]))
                        alpha.append(merged_score)
                    return alpha
            
            elif self.joint_merge_cfg.joint_method.standards == 'middle':
                window_left_size = self.joint_merge_cfg.joint_method.window_left_size
                window_right_size = window_size - window_left_size - 1
                if self.joint_merge_cfg.joint_method.strategy == 'continuous':
                    transducer_score,ctc_score = self.padding_continuous(transducer_score, ctc_score)

                for i in range(len(transducer_score)): 
                    left = max(0, i-window_left_size)
                    right = min(len(transducer_score)-1,i+window_right_size)
                    window_ctc = ctc_score[left:right]
                    window_trans = transducer_score[left:right]

                    similarity = 0
                    if self.joint_merge_cfg.joint_method.strategy == 'base' or self.joint_merge_cfg.joint_method.strategy == 'continuous':
                        similarity = self.cosine_similarity(window_ctc, window_trans)
                    elif self.joint_merge_cfg.joint_method.strategy == 'remove':
                        similarity = self.cosine_similarity_remove(window_ctc, window_trans)
                    
                    
                    
                    weight = max(0, similarity)
                    if transducer_score[i] == 'placeholder' and ctc_score[i] == 'placeholder':
                        alpha.append('placeholder')
                    elif transducer_score[i] == 'placeholder' and ctc_score[i] != 'placeholder':
                        fused_score = torch.exp(ctc_score[i])
                        alpha.append(fused_score)
                    elif transducer_score[i] != 'placeholder' and ctc_score[i] == 'placeholder':
                        fused_score = torch.exp(transducer_score[i])
                        alpha.append(fused_score)
                    else:
                        fused_score = (torch.exp(transducer_score[i]) + weight * torch.exp(ctc_score[i])) / (1 + weight)
                        alpha.append(fused_score)
                return alpha

                
            else:
                logger.warning('joint_method.standards is setting `{}` but not support'.format(self.joint_merge_cfg.joint_method.standards))
                import pdb;pdb.set_trace()
        else:
            logger.warning('joint_method is setting `{}` but not support'.format(self.joint_merge_cfg.joint_method.operation))
            import pdb;pdb.set_trace()
    
    # element-wise merge, which may be upgraded
    def _one_step_score_merge(self, transducer_score, ctc_score):
        if self.joint_merge_cfg.joint_method.operation == 'add':
            transducer_weight = self.joint_merge_cfg.joint_method.transducer_weight
            ctc_weight = 1 - transducer_weight

            merged_score = torch.log(transducer_weight * torch.exp(transducer_score) + ctc_weight * torch.exp(ctc_score))

            return merged_score
        else:
            logger.warning('joint_method is setting `{}` but not support'.format(self.joint_method))
            import pdb;pdb.set_trace()