from typing import List
import numpy as np
import torch
from scipy import special
from transformers import LlamaTokenizer

def inv_gumbel_cdf(xi, mu=0, beta=1, eps=1e-20):
    return mu - beta * np.log(-np.log(xi + eps))

class WmDetector:
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max: int = 0):
        
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.seed = seed
        self.shift_max = shift_max
        self.rng = torch.Generator(device='cpu')

    def get_szp_by_t(self, text:str, eps=1e-200):
        """
        Get p-value for each text.
        Args:
            text: a str, the text to detect
        Output:
            shift, zscore, pvalue, ntokens
        """
        pass 

class NgramWmDetector(WmDetector):
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            seed: int = 42,
            shift_max: int = 0,
            ngram: int = 1,
            seeding: str = 'hash',
            hash_key: int = 35317,
            scoring_method: str = "none",
        ):
        super().__init__(tokenizer, seed, shift_max)
        # watermark config
        self.ngram = ngram
        self.seeding = seeding
        self.hash_key = hash_key
        self.hashtable = np.random.permutation(1000003)
        self.scoring_method = scoring_method

    def hashint(self, integer_array):
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_array % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.hash_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.hash_key * np.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.hash_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.hash_key * input_ids)
            seed = np.min(seed)
        return seed

    def aggregate_scores(self, scores: List[np.array], aggregation: str = 'mean'):
        """Aggregate scores along a text."""
        scores = np.asarray(scores) # seq_len * (shift_max+1)
        if aggregation == 'sum':
            return scores.sum(axis=0)
        elif aggregation == 'mean':
            return scores.mean(axis=0) if scores.shape[0]!=0 else np.ones(shape=(self.vocab_size))
        elif aggregation == 'max':
            return scores.max(axis=0)
        else:
            raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(self, text: str, toks: int):
        """
        Get score increment for each token in a text
        Args:
            text: a text
            scoring_method: 
                'none': score all ngrams
                'v1': only score toksns for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
        Output:
            score: [np array of score increments for every token and payload] for a text
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_len = len(token_ids) if toks is None else min(len(token_ids), toks+4)
        start_pos = self.ngram +1
        rts = []
        seen_ntuples = set()
        for cur_pos in range(start_pos, total_len):
            ngram_tokens = token_ids[cur_pos-self.ngram:cur_pos] 
            if self.scoring_method == 'v1':
                tup_for_unique = tuple(ngram_tokens)
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            elif self.scoring_method == 'v2':
                tup_for_unique = tuple(ngram_tokens + token_ids[cur_pos:cur_pos+1])
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            rt = self.score_tok(ngram_tokens, token_ids[cur_pos]) 
            rt = rt[:self.shift_max+1]
            rts.append(rt)  
        return rts

    def get_szp_by_t(self, text: str, toks=None, eps=1e-200):
        ptok_scores = self.get_scores_by_t(text, toks)
        ptok_scores = np.asarray(ptok_scores) # ntoks x (shift_max+1)
        ntoks = ptok_scores.shape[0]
        aggregated_scores = ptok_scores.sum(axis=0) if ntoks!=0 else np.zeros(shape=ptok_scores.shape[-1]) # shift_max+1
        pvalues = [self.get_pvalue(score, ntoks, eps=eps) for score in aggregated_scores] # shift_max+1
        zscores = [self.get_zscore(score, ntoks) for score in aggregated_scores] # shift_max+1
        pvalue = min(pvalues)
        shift = pvalues.index(pvalue)
        zscore = zscores[shift]
        return int(shift), float(zscore), float(pvalue), ntoks
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the unit score """
        raise NotImplementedError
    
    def get_zscore(self, score: int, ntoks: int):
        """ compute the zscore from the total score and the number of tokens """
        raise NotImplementedError
    
    def get_pvalue(self, score: int, ntoks: int, eps: float=1e-200):
        """ compute the p-value from the total score and the number of tokens """
        raise NotImplementedError



class GumbelSoftDetectorNg(NgramWmDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        xi = torch.rand(self.vocab_size, generator=self.rng).numpy()
        xi = inv_gumbel_cdf(xi)
        scores = np.roll(xi, -token_id)
        return scores
    
    def get_zscore(self, score, ntoks):
        mu = 0.57721
        sigma = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu) / (sigma / np.sqrt(ntoks))
        return zscore
    
    def get_pvalue(self, score, ntoks, eps=1e-200):
        """ from cdf of a normal distribution """
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)