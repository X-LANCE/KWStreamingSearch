import torch
import torch.nn as nn

class FusionStrategy(nn.Module):
    def __init__(self, fusion_name, fusion_conf):
        super().__init__()
        self.fusion_name = fusion_name
        self.fusion_conf = fusion_conf

        self._call_dict = {
            'trans-dom': self.trans_dom,
            'ctc-dom': self.ctc_dom,
            'equ-dom': self.equ_dom,
            'cdc-zero': self.cdc_zero,
            'cdc-last': self.cdc_last,
        }
    
    def forward(self, trans_scores: list, ctc_scores: list):
        assert len(trans_scores) == len(ctc_scores)
        return self._call_dict[self.fusion_name](trans_scores, ctc_scores)

    def cdc_zero(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            left = max(0, t - self.fusion_conf['l_his'])        # l_his: history window size
            right = min(T - 1, t + self.fusion_conf['l_fut'])   # l_fut: future window size

            w_cdc = self.cosine_similarity(ctc_scores[left:right], trans_scores[left:right])
            fused_score = (torch.exp(trans_scores[t]) + w_cdc * torch.exp(ctc_scores[t])) / (1 + w_cdc)
            
            scores.append(fused_score)

            return scores

    def cdc_last(self, trans_scores, ctc_scores):
        trans_scores = self.padding_continuous(trans_scores)
        ctc_scores = self.padding_continuous(ctc_scores)

        return self.cdc_zero(trans_scores, ctc_scores)

    def trans_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if trans_scores[t] != 'placeholder':
                scores.append(trans_scores[t])
            else:
                scores.append(ctc_scores[t])
        return scores

    def ctc_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if ctc_scores[t] != 'placeholder':
                scores.append(ctc_scores[t])
            else:
                scores.append(trans_scores[t])
        return scores

    def equ_dom(self, trans_scores, ctc_scores):
        scores = []
        T = len(trans_scores)
        for t in range(T): 
            if trans_scores[t] == 'placeholder' and ctc_scores[t] == 'placeholder':
                scores.append('placeholder')
            elif trans_scores[t] != 'placeholder' and ctc_scores[t] != 'placeholder':
                merged_score = (trans_scores[t] + ctc_scores[t]) / 2
                scores.append(merged_score)
            elif trans_scores[t] == 'placeholder':
                scores.append(ctc_scores[t])
            else:
                scores.append(trans_scores[t])

        return scores

    def padding_scores(self, scores: list):
        T = len(scores)
        prev = torch.tensor(-1.0000e+35)
        for t in range(T):
            if scores[t] != 'placeholder':
                if self.fusion_name == 'cdc-last':
                    prev = scores[t]
                else:
                    prev = torch.tensor(-1.0000e+35)
            else:
                scores[t] = prev
        return scores

    # calculate the cosine_similarity of list[vec1] and list[vec2]
    def cosine_similarity(self, vec1, vec2):
        vec1 = [torch.tensor(-1.0000e+35) if x == 'placeholder' else x for x in vec1]
        vec2 = [torch.tensor(-1.0000e+35) if x == 'placeholder' else x for x in vec2]

        dot_product = sum(torch.exp(a) * torch.exp(b) for a, b in zip(vec1, vec2))
        norm_vec1 = torch.sqrt(sum(torch.exp(a) * torch.exp(a) for a in vec1))
        norm_vec2 = torch.sqrt(sum(torch.exp(b) * torch.exp(b) for b in vec2))
        
        return dot_product / (norm_vec1 * norm_vec2 + 1e-30)
