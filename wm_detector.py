"""
A wrapper class for watermark detector.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

# from torch._C import LongTensor
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


@dataclass
class DetectResult:
    z_score: float
    prediction: bool


####################
#                  #
#    Base class    #
#                  #
####################
class WMDetectorBase(ABC):
    """
    Abstract base class for watermark detector.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def detect_text(self, text: str, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given text.

        Args:
            text (str): text to be detected.
        """
        raise NotImplementedError

    @abstractmethod
    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        raise NotImplementedError

    def _state_dict(self) -> dict[str, Any]:
        return {
            "model_class_name": self.model.__class__.__name__,
            "tokenizer_class_name": self.tokenizer.__class__.__name__,
            "model_type_name": self.model.config.model_type,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.__class__ == WMDetectorBase:
            return self._state_dict()
        else:
            state: dict[str, Any] = super().state_dict()
            state.update(self._state_dict())
            return state

    @staticmethod
    def prepare_unbatched_input(input_ids: torch.LongTensor) -> torch.LongTensor:
        if input_ids.dim() == 1:
            # not batched
            return input_ids
        elif input_ids.dim() == 2:
            # batched
            assert input_ids.size(0) == 1
            return input_ids.squeeze(0)
        else:
            raise ValueError("input_ids must be 1D or 2D tensor.")


#############
#           #
#    KGW    #
#           #
#############
class KGWWMDetector(WMDetectorBase):
    """
    Wrapper class for KGW watermark detector.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        gamma: float,
        seeding_scheme: str,
        hash_key: int,
        *args,
        z_threshold: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, *args, **kwargs)
        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key
        self.z_threshold = z_threshold

        from watermarking.extended_watermark_processor import WatermarkDetector

        self.watermark_detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.gamma,  # should match original setting
            seeding_scheme=self.seeding_scheme,  # should match original setting
            device=self.model.device,  # must match the original rng device type
            tokenizer=self.tokenizer,
            z_threshold=self.z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
            hash_key=self.hash_key,
        )

    def detect_text(self, text: str, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        raw_score = self.watermark_detector.detect(text)
        return DetectResult(z_score=raw_score["z_score"], prediction=raw_score["prediction"])

    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        ids = self.prepare_unbatched_input(input_ids)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.detect_text(text)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "seeding_scheme": self.seeding_scheme,
            "hash_key": self.hash_key,
            "z_threshold": self.z_threshold,
        }


#############
#           #
#    SIR    #
#           #
#############
class SIRWMDetector(WMDetectorBase):
    """
    Wrapper class for SIR watermark detector.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        window_size: int,
        gamma: float,
        delta: float,
        hash_key: int,
        z_threshold: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, *args, **kwargs)
        self.window_size = window_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.z_threshold = z_threshold

        from robust_watermark.watermark import WatermarkWindow

        self.watermark_detector = WatermarkWindow(
            device=self.model.device,
            window_size=self.window_size,
            gamma=self.gamma,
            delta=self.delta,
            target_tokenizer=self.tokenizer,
            hash_key=self.hash_key,
        )

    def detect_text(self, input_text):
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        raw_score = self.watermark_detector.detect(input_text)
        prediction_result = True if raw_score > self.z_threshold else False
        return DetectResult(z_score=raw_score, prediction=prediction_result)

    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        ids = self.prepare_unbatched_input(input_ids)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.detect_text(text)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "gamma": self.gamma,
            "delta": self.delta,
            "hash_key": self.hash_key,
            "z_threshold": self.z_threshold,
        }
