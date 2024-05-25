import torch
import numpy as np
import random
import time
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration

from typing import List

def inv_gumbel_cdf(xi, mu=0, beta=1, eps=1e-20):
    return mu - beta * torch.log(-torch.log(xi + eps))

class Generator():
    """
    This is a general purpose generator, all other generator is a subclass of this generator
    """
    def __init__(self,
            model: LlamaForCausalLM,
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max: int = 0,
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.seed = seed
        self.shift_max = shift_max

        self.max_seq_len = model.config.max_position_embeddings
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        self.eos_id = model.config.eos_token_id
        self.ngram = 0
        #for generator we use it to sample a torch.tensor
        self.rng = torch.Generator(device='cpu')
        self.device = model.device

    @torch.no_grad()
    def generate(
        self,
        # prompts: List[str],
        input_ids: torch.LongTensor,
        max_gen_len: int,
        temperature: float = 1,
        top_p: float = 1,
    ) -> List[str]:
        """
        Generate text from prompts. 
        For each call to generate, we deem it as a response, and we assign an(almost) unique identifier for each response.
        Adapted from https://github.com/facebookresearch/llama/
        """
        # bsz = len(prompts)
        # prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        prompt_tokens = input_ids
        bsz = prompt_tokens.shape[0]
        assert bsz == 1, "Batch size should be 1"
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.pad_id).to(self.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        
        unique_identifier = self.gen_unique_id(bsz)
        for cur_pos in range(start_pos, total_len):
            # Use LLM to calculate logits vector l
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            xi = self.F_key(unique_identifier, ngram_tokens, cur_pos-start_pos)
            next_toks = self.Gamma(xi, outputs.logits[:, -1, :], temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos
        tk = tokens[0].tolist()
        tk = tk[:len(input_ids[0]) + max_gen_len]
        if tk.count(self.eos_id) != 0:
            tk = tk[: tk.index(self.eos_id)]
        # Match output format..
        return torch.LongTensor([tk])
        
    def gen_unique_id(self, bsz):
        """ Generate a unique identifier for each response """
        return np.random.randint(self.shift_max+1, size=bsz)
    
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t, we use identifier r and position t/ngram
        to calculate a watermark key."""
        pass
    
    def top_p(self, logits, temperature, top_p):
        """ An utility function for top_p sampling """
        probs = torch.softmax(logits / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort >= top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token_probs = torch.zeros(logits.shape).to(self.device).scatter_(-1, probs_idx, probs_sort) # probability of next token, ordered by vocab 
        return next_token_probs
    
    def Gamma(
        self,
        xi, # the watermark key for current position
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:
        """ This is the decoder: Take a watermark key xi and a logits vector l to decide the next token.
        Vanilla sampling with temperature and top p."""
        if temperature > 0:
            next_token_probs = self.top_p(logits, temperature, top_p)  
            next_token = torch.multinomial(next_token_probs, num_samples=1) # one hot of next token, ordered by original probs
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class NgramWmGenerator(Generator):
    """
    This kind of generator use previous Ngram and a hash function to calculate the watermark key.
    """
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer,
            seed: int = 42,
            shift_max : int = 0, 
            ngram: int = 1,
            seeding: str = 'hash',
            hash_key: int = 35317,
        ):
        # model config
        super().__init__(model, tokenizer, seed, shift_max)
        # watermark config
        self.ngram = ngram
        self.seeding = seeding 
        self.hash_key = hash_key
        self.hashtable = torch.randperm(1000003)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.hash_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.hash_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.hash_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.hash_key * input_ids)
            seed = torch.min(seed).item()
        return seed


class GumbelSoftGeneratorNg(NgramWmGenerator):
    """ Generate text using LM and Gumbel-softmax watermarking method. """
    def __init__(self, 
            *args, 
            drop_prob = 0,
            tau=0,
            **kwargs,
        ):
        super().__init__(*args, **kwargs) 
        self.drop_prob = drop_prob/100
        self.tau=tau
        
    def F_key(self, r, ngram, t):
        """ calculate the watermark key xi at position t """
        bsz = ngram.shape[0]
        batched_xi = []
        for i in range(bsz):
            seed = self.get_seed_rng(ngram[i])
            self.rng.manual_seed(seed)
            xi = torch.rand(self.vocab_size, generator=self.rng)
            xi = inv_gumbel_cdf(xi)
            xi = xi.roll(-r[i])
            batched_xi.append(xi)
            
        return batched_xi 
     
    def Gamma(
        self,
        xis,  # a list of 'bsz' xis
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
    ) -> torch.LongTensor:  
        if temperature > 0:
            probs = self.top_p(logits, temperature, top_p)
            xis = torch.stack(xis).to(self.device)
            if self.tau>0:
                # run Gumbel-softmax
                next_token =  torch.multinomial(torch.softmax((probs.log()+xis)/self.tau, dim=-1), num_samples=1) 
            else:
                # run Logp-Addition, drop probability is self.drop_prob
                if np.random.rand()<self.drop_prob:
                    # sample next token based on original probability distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs.log()+xis, dim=-1, keepdim=True)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
