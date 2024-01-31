"""
A wrapper class for watermark detector.
"""

import dataclasses
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Type

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedTokenizer


def get_detector_class_from_type(type: str) -> Type["WMDetectorBase"]:
    match type:
        case "KGW":
            return KGWWMDetector
        case "SIR":
            return SIRWMDetector
        case "UBW":
            warnings.warn(f"UBW is not suitable for paraphrasing attacks.", DeprecationWarning)
            return UBWWMDetector
        case "PRW":
            return PRWWMDetector
        case "RDW":
            return RDWWMDetector
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
                elif isinstance(v, np.bool_):
                    ret[k] = bool(v)
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

        from KGW.extended_watermark_processor import WatermarkDetector

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
        z_threshold: float = 0.0,
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
        self.z_threshold = z_threshold

        from SIR.watermark import WatermarkContext, WatermarkWindow

        if mode == "window":
            self.watermark_detector = WatermarkWindow(
                device=self.model.device,
                window_size=self.window_size,
                target_tokenizer=self.tokenizer,
                gamma=self.gamma,
                delta=self.delta,
                hash_key=self.key,
            )
        elif mode == "context":
            self.watermark_detector = WatermarkContext(
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

    @torch.no_grad()
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
        temperature: float = 1.0,
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
        self.temperature = temperature
        self.ctx_n = ctx_n
        # process pool for scorer
        from concurrent.futures import ProcessPoolExecutor

        self.process_pool = ProcessPoolExecutor(max_workers=8)

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

    @torch.no_grad()
    def detect_text(self, text: str, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        from unbiased_watermark import LLR_Score, RobustLLR_Score, get_score

        # NOTE: Hyperparameters are fixed for now.
        scorer = RobustLLR_Score(0.1, 0.1, process_pool=self.process_pool)
        # scorer = LLR_Score()

        raw_score, _prompt_len = get_score(
            text,
            watermark_processor=self.warper,
            score=scorer,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.temperature,
            prompt="",
        )

        score = torch.clamp_(raw_score, -100, 100)

        return DetectResult(llr_score=float(score.mean().abs().item()))

    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        input_ids = self.prepare_unbatched_input(input_ids)
        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return self.detect_text(text, *args, **kwargs)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "gamma": self.gamma,
            "temperature": self.temperature,
            "ctx_n": self.ctx_n,
        }


#############
#           #
#    RDW    #
#           #
#############
class RDWWMDetector(WMDetectorBase):
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

    def detect(self, tokens, n, k, xi, gamma=0.0):
        import pyximport

        pyximport.install(
            reload_support=True,
            language_level=sys.version_info[0],
            setup_args={"include_dirs": np.get_include()},
        )
        from RDW.levenshtein import levenshtein

        m = len(tokens)
        n = len(xi)

        A = np.empty((m - (k - 1), n))
        for i in range(m - (k - 1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i : i + k], xi[(j + np.arange(k)) % n], gamma)

        return np.min(A)

    @torch.no_grad()
    def detect_text(self, input_text):
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        raise NotImplementedError

    @torch.no_grad()
    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        input_ids = self.prepare_unbatched_input(input_ids).numpy()
        token_length = len(input_ids)
        vocab_size = len(self.tokenizer)
        n_runs = 100
        from RDW.mersenne import mersenne_rng

        rng = mersenne_rng(self.key)
        xi = np.array(
            [rng.rand() for _ in range(self.wm_sequence_length * vocab_size)], dtype=np.float32
        ).reshape(self.wm_sequence_length, vocab_size)
        test_result = self.detect(input_ids, self.wm_sequence_length, token_length, xi)

        p_val = 0
        for _ in range(n_runs):
            xi_alternative = np.random.rand(self.wm_sequence_length, vocab_size).astype(np.float32)
            null_result = self.detect(
                input_ids, self.wm_sequence_length, token_length, xi_alternative
            )

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result

        p_value = (p_val + 1.0) / (n_runs + 1.0)
        return DetectResult(z_score=p_value)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "wm_sequence_length": self.wm_sequence_length,
        }


#######################
#                     #
#    Unigram / PRW    #
#                     #
#######################
class PRWWMDetector(WMDetectorBase):
    """
    Wrapper class for Unigram watermark detector.
    """

    TYPE = "PRW"

    def __init__(
        self,
        model: AutoModelForCausalLM | Any,
        tokenizer: PreTrainedTokenizer | Any,
        key: int,
        *args,
        z_threshold: float = 6.0,
        fraction: float = 0.5,
        strength: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, key, *args, **kwargs)
        self.fraction = fraction
        self.strength = strength
        self.z_threshold = z_threshold

        from unigram_watermark.gptwm import GPTWatermarkDetector

        self.watermark_detector = GPTWatermarkDetector(
            fraction=self.fraction,
            strength=self.strength,
            vocab_size=self.tokenizer.vocab_size,
            watermark_key=int(self.key),
        )

    def detect_text(self, text: str, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return self.detect_tokens(ids)

    def detect_tokens(self, input_ids: torch.LongTensor, *args, **kwargs) -> DetectResult:
        """
        Detect watermark given input_ids.

        Args:
            input_ids (torch.LongTensor): input_ids to be detected.
        """
        ids = self.prepare_unbatched_input(input_ids)
        raw_score = self.watermark_detector.detect(ids.flatten().tolist())
        return DetectResult(z_score=raw_score, prediction=raw_score > self.z_threshold)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "strength": self.strength,
            "z_threshold": self.z_threshold,
        }
