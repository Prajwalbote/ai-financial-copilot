"""
llm/base_llm.py — LLM Wrapper
==============================
WHY FLAN-T5?
    - Completely free, no API costs
    - Runs on CPU (no GPU needed)
    - Good at instruction following
    - 250M parameters — small but capable
    - Perfect for learning RAG concepts

    In production you'd swap this for:
    - Llama 3 (better quality, needs GPU)
    - Mistral 7B (great quality/speed tradeoff)
    - GPT-4 via API (best quality, costs money)

    Our wrapper makes swapping models easy —
    change one line in config.yaml.

WHAT YOU LEARN:
    - How to load and run HuggingFace LLMs
    - Text generation parameters (temperature, etc.)
    - Why we separate LLM from RAG logic
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class FinancialLLM:
    """
    Wrapper around Flan-T5 for financial question answering.

    WHY LAZY LOADING AGAIN?
    LLMs are large. Flan-T5-base is 900MB.
    We only load it when actually needed.

    Usage:
        llm = FinancialLLM()
        answer = llm.generate("What is Apple's revenue?", context)
    """

    def __init__(self):
        self.cfg = get_config()
        self.model_name = self.cfg.llm.model_name
        self.device = self.cfg.llm.device
        self.max_input_tokens = self.cfg.llm.max_input_tokens
        self.max_new_tokens = self.cfg.llm.max_new_tokens

        # Lazy loaded
        self._model = None
        self._tokenizer = None

        log.info(
            f"FinancialLLM configured | "
            f"model={self.model_name} | "
            f"device={self.device}"
        )

    def _load_model(self):
        """Load model and tokenizer on first use."""
        if self._model is None:
            log.info(f"Loading LLM: {self.model_name}")
            log.info("First load takes 1-2 minutes (downloading ~900MB)...")

            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_name
            )

            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )

            self._model.to(self.device)
            self._model.eval()  # Set to inference mode (no gradient tracking)

            log.info(f"LLM loaded successfully | device={self.device}")

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Full prompt including context and question

        Returns:
            Generated text response

        KEY PARAMETERS EXPLAINED:
        - max_new_tokens: Max length of generated answer
        - num_beams: Beam search width (higher = better but slower)
          Beam search keeps multiple candidate sequences and
          picks the best one — better than greedy decoding
        - early_stopping: Stop when all beams hit end token
        - no_repeat_ngram_size: Prevents repetitive text
        """
        self._load_model()

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,        # Cut if prompt exceeds max_length
            padding=True,
        ).to(self.device)

        log.info(
            f"Generating response | "
            f"input_tokens={inputs['input_ids'].shape[1]}"
        )

        # Generate with no gradient computation (saves memory)
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        # Decode token IDs back to text
        response = self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True   # Remove <pad>, </s> tokens
        )

        log.info(f"Generated response | length={len(response)} chars")
        return response.strip()