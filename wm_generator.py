"""
A wrapper class for watermark generator.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Type

import torch
from transformers import GenerationMixin, LogitsProcessorList, PreTrainedTokenizer


def get_generator_class_from_type(type: Literal["KGW", "SIR", "UBW"]) -> Type["WMGeneratorBase"]:
    match type:
        case "KGW":
            return KGWWMGenerator
        case "SIR":
            return SIRWMGenerator
        case "UBW":
            return UBWWMGenerator
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

        from watermarking.extended_watermark_processor import WatermarkLogitsProcessor

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
            "hash_key": self.key,
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
        chunk_size: int,
        transform_model_path: str,
        embedding_model: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.window_size = window_size
        self.gamma = gamma
        self.delta = delta
        self.chunk_size = chunk_size
        self.transform_model_path = os.path.join(os.path.dirname(__file__), transform_model_path)
        self.embedding_model = embedding_model

        from robust_watermark.watermark import (
            WatermarkContext,
            WatermarkLogitsProcessor,
            WatermarkWindow,
        )

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
                chunk_size=self.chunk_size,
                tokenizer=self.tokenizer,
                gamma=self.gamma,
                delta=self.delta,
                transform_model_path=self.transform_model_path,
                embedding_model=self.embedding_model,
            )
        else:
            raise
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
            "hash_key": self.key,
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
            "private_key": self.key,
            "gamma": self.gamma,
            "ctx_n": self.ctx_n,
        }
