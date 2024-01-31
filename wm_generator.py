"""
A wrapper class for watermark generator.
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Type

import torch
from transformers import GenerationMixin, LogitsProcessorList, PreTrainedTokenizer


def get_generator_class_from_type(type: str) -> Type["WMGeneratorBase"]:
    match type:
        case "KGW":
            return KGWWMGenerator
        case "SIR":
            return SIRWMGenerator
        case "UBW":
            warnings.warn(f"UBW is not suitable for paraphrasing attacks.", DeprecationWarning)
            return UBWWMGenerator
        case "PRW":
            return PRWWMGenerator
        case "RDW":
            return RDWWMGenerator
        case _:
            raise ValueError(f"Invalid type: {type}")


####################
#                  #
#    Base class    #
#                  #
####################
class WMGeneratorBase(ABC):
    """
    Abstract base class for watermark generator.
    """

    TYPE = "base"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: Any,
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.key = key

    @abstractmethod
    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        raise NotImplementedError

    def tokens2text(self, input_ids: torch.LongTensor, *args, **kwargs) -> str:
        """
        Convert input_ids to text. Automatically handle batched or non-batched input_ids.
        """
        if input_ids.dim() == 1:
            # not batched
            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
        elif input_ids.dim() == 2:
            # batched
            assert input_ids.size(0) == 1, "batch size must be 1."
            return self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        else:
            raise ValueError("input_ids must be 2D or 3D tensor.")

    def _state_dict(self) -> dict[str, Any]:
        return {
            "model_class_name": self.model.__class__.__name__,
            "tokenizer_class_name": self.tokenizer.__class__.__name__,
            "model_type_name": self.model.config.model_type,
            "key": self.key,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.__class__ == WMGeneratorBase:
            return self._state_dict()
        else:
            state: dict[str, Any] = super().state_dict()
            state.update(self._state_dict())
            return state

    @staticmethod
    def prepare_batched_input(input_ids: torch.LongTensor) -> torch.LongTensor:
        if input_ids.dim() == 1:
            # not batched
            return input_ids.unsqueeze(0)
        elif input_ids.dim() == 2:
            # batched
            return input_ids
        else:
            raise ValueError("input_ids must be 1D or 2D tensor.")


#############
#           #
#    KGW    #
#           #
#############
class KGWWMGenerator(WMGeneratorBase):
    """
    Wrapper class for KGW watermark generator.
    Ref:
        Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I. & Goldstein, T.. (2023). A Watermark for Large Language Models.
        Proceedings of the 40th International Conference on Machine Learning, in Proceedings of Machine Learning Research 202:17061-17084.
    """

    TYPE = "KGW"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: Any,
        gamma: float,
        delta: float,
        seeding_scheme: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme

        from KGW.extended_watermark_processor import WatermarkLogitsProcessor

        self.logits_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            hash_key=self.key,
        )

    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        input_ids = self.prepare_batched_input(input_ids)
        # generate watermark tokens
        output_tokens = self.model.generate(
            input_ids,
            *args,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **kwargs,
        )

        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        if truncate_output:
            output_tokens = output_tokens[:, input_ids.shape[-1] :]

        return output_tokens

    def _state_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "delta": self.delta,
            "seeding_scheme": self.seeding_scheme,
        }


#############
#           #
#    SIR    #
#           #
#############
class SIRWMGenerator(WMGeneratorBase):
    """
    Wrapper class for SIR watermark generator.
    Ref:
        Li, Y., Wang, X., Wang, Z., Zhang, Y., Liu, Q., & Wang, H. (2020). SIR: A Sentiment-Infused Recommender System.
        arXiv preprint arXiv:2008.13535.
    """

    TYPE = "SIR"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: int,
        mode: Literal["window", "context"],
        window_size: int,
        gamma: float,
        delta: float,
        chunk_length: int,
        transform_model_path: str,
        embedding_model: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.window_size = window_size
        self.gamma = gamma
        self.delta = delta
        self.chunk_length = chunk_length
        self.transform_model_path = os.path.join(
            os.path.dirname(__file__), "robust_watermark", transform_model_path
        )
        self.embedding_model = embedding_model

        from SIR.watermark import WatermarkContext, WatermarkLogitsProcessor, WatermarkWindow

        if mode == "window":
            self.watermark_model = WatermarkWindow(
                device=self.model.device,
                window_size=self.window_size,
                target_tokenizer=self.tokenizer,
                gamma=self.gamma,
                delta=self.delta,
                hash_key=self.key,
            )
        elif mode == "context":
            self.watermark_model = WatermarkContext(
                device=self.model.device,
                chunk_length=self.chunk_length,
                target_tokenizer=self.tokenizer,
                gamma=self.gamma,
                delta=self.delta,
                transform_model_path=self.transform_model_path,
                embedding_model=self.embedding_model,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        self.logits_processor = WatermarkLogitsProcessor(self.watermark_model)

    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        input_ids = self.prepare_batched_input(input_ids)
        # generate watermark tokens
        output_tokens = self.model.generate(
            input_ids,
            *args,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **kwargs,
        )

        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        if truncate_output:
            output_tokens = output_tokens[:, input_ids.shape[-1] :]

        return output_tokens

    def _state_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "gamma": self.gamma,
            "delta": self.delta,
        }


################
#              #
#    Unbias    #
#              #
################
class UBWWMGenerator(WMGeneratorBase):
    """
    Wrapper class for unbiased watermark generator.
    Ref:
        Unbiased Watermark for Large Language Models. https://arxiv.org/abs/2310.10669
    """

    TYPE = "UBW"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: Any,
        mode: Literal["delta", "gamma"],
        *args,
        gamma: float = 1.0,
        ctx_n: int = 5,
        **kwargs,
    ) -> None:
        """
        Args:
            key: Must satisfy the Buffer API, like bytes objects.
        """
        key = bytes(str(key), encoding="utf-8")
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.mode = mode
        self.gamma = gamma
        self.ctx_n = ctx_n

        from unbiased_watermark import (
            Delta_Reweight,
            Gamma_Reweight,
            PrevN_ContextCodeExtractor,
            WatermarkLogitsProcessor,
            patch_model,
        )

        # current generation() doesn't accept logits_warper parameter.
        patch_model(self.model)

        if self.mode == "delta":
            self.warper = WatermarkLogitsProcessor(
                self.key,
                Delta_Reweight(),
                PrevN_ContextCodeExtractor(self.ctx_n),
            )
        elif self.mode == "gamma":
            self.warper = WatermarkLogitsProcessor(
                self.key,
                Gamma_Reweight(self.gamma),
                PrevN_ContextCodeExtractor(self.ctx_n),
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        input_ids = self.prepare_batched_input(input_ids)
        # generate watermark tokens
        output_tokens = self.model.generate(
            input_ids,
            *args,
            logits_warper=LogitsProcessorList([self.warper]),
            **kwargs,
        )

        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        if truncate_output:
            output_tokens = output_tokens[:, input_ids.shape[-1] :]

        return output_tokens

    def _state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "gamma": self.gamma,
            "ctx_n": self.ctx_n,
        }


#############
#           #
#    RDW    #
#           #
#############
class RDWWMGenerator(WMGeneratorBase):
    """
    Wrapper class for unbiased watermark generator.
    Ref:
        Unbiased Watermark for Large Language Models. https://arxiv.org/abs/2310.10669
    """

    TYPE = "RDW"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: Any,
        wm_sequence_length: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.wm_sequence_length = wm_sequence_length

    def generate_shift(
        self,
        model: GenerationMixin | Any,
        input_ids: torch.LongTensor,
        vocab_size: int,
        wm_sequence_length: int,
        max_new_tokens: int,
        key: int,
        **kwargs,
    ):
        from RDW.mersenne import mersenne_rng

        rng = mersenne_rng(key)
        xi = torch.tensor([rng.rand() for _ in range(wm_sequence_length * vocab_size)]).view(
            wm_sequence_length, vocab_size
        )
        shift = torch.randint(wm_sequence_length, (1,))

        inputs = input_ids.to(model.device)
        attn = torch.ones_like(inputs)
        past = None
        for i in range(max_new_tokens):
            with torch.no_grad():
                if past:
                    output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = model(inputs)

            probs = torch.nn.functional.softmax(output.logits[:, -1, :vocab_size], dim=-1).cpu()
            token = (
                torch.argmax(xi[(shift + i) % wm_sequence_length, :] ** (1 / probs), axis=1)
                .unsqueeze(-1)
                .to(model.device)
            )
            inputs = torch.cat([inputs, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return inputs.detach().cpu()

    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        input_ids = self.prepare_batched_input(input_ids)
        # generate watermark tokens
        output_tokens = self.generate_shift(
            self.model,
            input_ids,
            vocab_size=len(self.tokenizer),
            wm_sequence_length=self.wm_sequence_length,
            key=self.key,
            **kwargs,
        )

        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        if truncate_output:
            output_tokens = output_tokens[:, input_ids.shape[-1] :]

        return output_tokens

    def _state_dict(self) -> dict[str, Any]:
        return {
            "wm_sequence_length": self.wm_sequence_length,
        }


#######################
#                     #
#    Unigram / PRW    #
#                     #
#######################
class PRWWMGenerator(WMGeneratorBase):
    """
    Wrapper class for Unigram watermark generator. https://arxiv.org/abs/2306.17439
    Ref:
        Zhao, X., Ananth, P., Li, L., & Wang, Y. X. (2023). Provable robust watermarking for ai-generated text. arXiv preprint arXiv:2306.17439.
    """

    TYPE = "PRW"

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: int,
        *args,
        fraction: float = 0.5,
        strength: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.fraction = fraction
        self.strength = strength

        from unigram_watermark.gptwm import GPTWatermarkLogitsWarper

        self.logits_processor = GPTWatermarkLogitsWarper(
            fraction=self.fraction,
            strength=self.strength,
            vocab_size=self.tokenizer.vocab_size,
            watermark_key=int(self.key),
        )

    def generate(
        self, input_ids: torch.LongTensor, *args, truncate_output: bool = True, **kwargs
    ) -> torch.LongTensor:
        """
        Generate watermark tokens given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be watermarked.
            truncate_output (bool): whether to truncate the output to the newly created tokens.
        """
        input_ids = self.prepare_batched_input(input_ids)
        # generate watermark tokens
        output_tokens = self.model.generate(
            input_ids,
            *args,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **kwargs,
        )

        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        if truncate_output:
            output_tokens = output_tokens[:, input_ids.shape[-1] :]

        return output_tokens

    def _state_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "strength": self.strength,
        }
