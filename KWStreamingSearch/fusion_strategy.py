import torch
import torch.nn as nn

EPS = 1e-10
PH = "placeholder"

"""
    fusion_types:
        - cdc-vanilla: average of CTC score and sliding window cosine similarity (of ICTC and CTC)
        ---------------------------------------------------------------------------------------------
        - trans-dom: Transducer domination
        - ctc-dom: CTC branch domination
        - equ-dom: equvilant domination
        - cdc-zero: consider placeholder as 0, followed by the adapted CDC fusion
        - cdc-last: pad placeholder as last non-zero score, followed by the adapted CDC fusion
        
    fusion_conf: this dict is only used when `fusion_type.find('cdc') >= 0`
        l_his: int, history window size (#frames)
        l_fut: int, future window size (#frames)
"""
class FusionStrategy(nn.Module):
    def __init__(self, fusion_name, fusion_conf={'l_his': 0, 'l_fut': 30}):
        super().__init__()
        self.fusion_name = fusion_name
        self.fusion_conf = fusion_conf

        self._call_dict = {
            'trans-dom': self.trans_dom,
            'ctc-dom': self.ctc_dom,
            'equ-dom': self.equ_dom,
            'cdc-zero': self.cdc_zero,
            'cdc-last': self.cdc_last,
            'cdc-vanilla': self.cdc_vanilla,
        }
    
    def forward(self, trans_scores: list, ctc_scores: list):
        assert len(trans_scores) == len(ctc_scores)
        return self._call_dict[self.fusion_name](trans_scores, ctc_scores)

    """
    Paper:
        title: Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency 
        arxiv: https://arxiv.org/pdf/2412.12635
        IEEE: https://ieeexplore.ieee.org/document/10890010 
    """
    def cdc_vanilla(self, ctc_scores: list, ictc_scores: list):
        scores = []
        T = len(ctc_scores)
        for t in range(T): 
            left = max(0, t - self.fusion_conf['l_his'])        # l_his: history window size
            right = min(T - 1, t + self.fusion_conf['l_fut'])   # l_fut: future window size

            s_cdc = self._cosine_similarity(ctc_scores[left:right], ictc_scores[left:right])
            s_refine = (ctc_scores[t] + s_cdc) / 2
            
            scores.append(s_refine)

            return scores

    """
    Paper:
        title: MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding  
    """
    def cdc_zero(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            left = max(0, t - self.fusion_conf['l_his'])        # l_his: history window size
            right = min(T - 1, t + self.fusion_conf['l_fut'])   # l_fut: future window size

            w_cdc = self._cosine_similarity(ctc_scores[left:right], trans_scores[left:right])
            fused_score = (trans_scores[t] + w_cdc * ctc_scores[t]) / (1 + w_cdc)
            
            scores.append(fused_score)

            return scores

    def cdc_last(self, trans_scores, ctc_scores):
        trans_scores = self._padding_scores(trans_scores)
        ctc_scores = self._padding_scores(ctc_scores)

        return self.cdc_zero(trans_scores, ctc_scores)

    def trans_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if trans_scores[t] != PH:
                scores.append(trans_scores[t])
            else:
                scores.append(ctc_scores[t])
        return scores

    def ctc_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if ctc_scores[t] != PH:
                scores.append(ctc_scores[t])
            else:
                scores.append(trans_scores[t])
        return scores

    def equ_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if trans_scores[t] == PH and ctc_scores[t] == PH:
                scores.append(PH)
            elif trans_scores[t] != PH and ctc_scores[t] != PH:
                merged_score = (trans_scores[t] + ctc_scores[t]) / 2
                scores.append(merged_score)
            elif trans_scores[t] == PH:
                scores.append(ctc_scores[t])
            else:
                scores.append(trans_scores[t])

        return scores

    '''
        pad placeholder to last non-zero score
    '''
    def _padding_scores(self, scores: list):
        T = len(scores)
        prev = torch.tensor(EPS)
        for t in range(T):
            if scores[t] != PH:
                if self.fusion_name == 'cdc-last':
                    prev = scores[t]
                else:
                    prev = torch.tensor(EPS)
            else:
                scores[t] = prev
        return scores

    '''
        cosine similarity
    '''
    def _cosine_similarity(self, vec1, vec2):
        vec1 = torch.tensor([torch.tensor(EPS) if x == PH else x for x in vec1])
        vec2 = torch.tensor([torch.tensor(EPS) if x == PH else x for x in vec2])

        # dot_product = (vec1 * vec2).sum() sum(a * b for a, b in zip(vec1, vec2))
        # norm_vec1 = torch.sqrt(sum(a * a for a in vec1))
        # norm_vec2 = torch.sqrt(sum(b * b for b in vec2))
        
        dot_product = (vec1 * vec2).sum()
        norm_vec1 = vec1.norm(2)
        norm_vec2 = vec2.norm(2)

        return dot_product / (norm_vec1 * norm_vec2 + EPS)
