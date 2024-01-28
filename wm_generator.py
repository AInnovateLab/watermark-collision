"""
A wrapper class for watermark generator.
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import GenerationMixin, LogitsProcessorList, PreTrainedTokenizer


####################
#                  #
#    Base class    #
#                  #
####################
class WMGeneratorBase(ABC):
    """
    Abstract base class for watermark generator.
    """

    def __init__(
        self, model: GenerationMixin | Any, tokenizer: PreTrainedTokenizer | Any, *args, **kwargs
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

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
        if input_ids.dim() == 2:
            # not batched
            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
        elif input_ids.dim() == 3:
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
        if input_ids.dim() == 2:
            # not batched
            return input_ids.unsqueeze(0)
        elif input_ids.dim() == 3:
            # batched
            return input_ids
        else:
            raise ValueError("input_ids must be 2D or 3D tensor.")


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

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        gamma: float,
        delta: float,
        seeding_scheme: str,
        hash_key: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(model, tokenizer, *args, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key

        from watermarking.extended_watermark_processor import WatermarkLogitsProcessor

        self.logits_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            hash_key=self.hash_key,
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
            "hash_key": self.hash_key,
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

    def __init__(
        self,
        model: GenerationMixin | Any,
        tokenizer: PreTrainedTokenizer | Any,
        window_size: int,
        gamma: float,
        delta: float,
        hash_key: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(model, tokenizer, *args, **kwargs)
        self.window_size = window_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

        from robust_watermark.watermark import WatermarkWindow

        # TODO
        raise NotImplementedError

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

    def _state_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "gamma": self.gamma,
            "delta": self.delta,
            "hash_key": self.hash_key,
        }
