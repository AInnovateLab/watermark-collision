"""
A wrapper class for watermark detector.
"""
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Type

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


def get_detector_class_from_type(type: Literal["KGW", "SIR", "UBW"]) -> Type["WMDetectorBase"]:
    match type:
        case "KGW":
            return KGWWMDetector
        case "SIR":
            return SIRWMDetector
        case "unbiased":
            return UBWWMDetector
        case _:
            raise ValueError(f"Invalid type: {type}")


@dataclass
class DetectResult:
    # KGW metrics
    z_score: float | None = None
    prediction: bool | None = None
    # Unbiased metrics
    llr_score: float | None = None

    def asdict(self) -> dict[str, Any]:
        def _to_dict(x):
            ret = {}
            for k, v in x:
                if v is None:
                    continue
                elif isinstance(v, float):
                    ret[k] = round(v, 4)
                else:
                    ret[k] = v
            return ret

        return dataclasses.asdict(self, dict_factory=_to_dict)


####################
#                  #
#    Base class    #
#                  #
####################
class WMDetectorBase(ABC):
    """
    Abstract base class for watermark detector.
    """

    TYPE = "base"

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: Any,
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.key = key

    @abstractmethod
    def detect_text(self, text: str, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given text.

        Args:
            text (str): text to be detected.
        """
        ...

    @abstractmethod
    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        ...

    def _state_dict(self) -> dict[str, Any]:
        return {
            "model_class_name": self.model.__class__.__name__,
            "tokenizer_class_name": self.tokenizer.__class__.__name__,
            "model_type_name": self.model.config.model_type,
            "key": self.key,
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

    TYPE = "KGW"

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: int,
        gamma: float,
        seeding_scheme: str,
        *args,
        z_threshold: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
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
            hash_key=self.key,
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
            "hash_key": self.key,
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

    TYPE = "SIR"

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: int,
        window_size: int,
        gamma: float,
        delta: float,
        z_threshold: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.window_size = window_size
        self.gamma = gamma
        self.delta = delta
        self.z_threshold = z_threshold

        from robust_watermark.watermark import WatermarkWindow

        self.watermark_detector = WatermarkWindow(
            device=self.model.device,
            window_size=self.window_size,
            gamma=self.gamma,
            delta=self.delta,
            target_tokenizer=self.tokenizer,
            hash_key=self.key,
        )

    def detect_text(self, input_text):
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        raw_score = self.watermark_detector.detect(input_text)
        prediction_result = raw_score > self.z_threshold
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
            "hash_key": self.key,
            "z_threshold": self.z_threshold,
        }


################
#              #
#    Unbias    #
#              #
################
class UBWWMDetector(WMDetectorBase):
    """
    Wrapper class for Unbiased watermark detector.
    Ref:
        Unbiased Watermark for Large Language Models. https://arxiv.org/abs/2310.10669
    """

    TYPE = "UBW"

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
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
        )

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

    def detect_text(self, text: str, temperature: float, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        from unbiased_watermark import RobustLLR_Score, get_score

        # NOTE: Hyperparameters are fixed for now.
        scorer = RobustLLR_Score(0.1, 0.1)

        raw_score, _prompt_len = get_score(
            text,
            watermark_processor=self.warper,
            score=scorer,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            prompt="",
        )

        return DetectResult(llr_score=raw_score)

    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        input_ids = self.prepare_unbatched_input(input_ids)
        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return self.detect_text(text)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "private_key": self.key,
            "gamma": self.gamma,
            "ctx_n": self.ctx_n,
        }
