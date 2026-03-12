"""
finetuning/evaluate.py — Model Evaluation
==========================================
WHY EVALUATE?
    Training loss going down is good but not enough.
    We need to know: does the model actually answer
    financial questions better than before?

WHAT YOU LEARN:
    - ROUGE score for text generation evaluation
    - Before/after comparison methodology
    - How to load and use a fine-tuned LoRA model
"""

import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from utils.logger import get_logger

log = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates base vs fine-tuned model on financial QA.

    Usage:
        evaluator = ModelEvaluator()
        evaluator.compare_models(test_questions)
    """

    # Test questions for evaluation
    TEST_QUESTIONS = [
        {
            "question": "What are Apple total net sales?",
            "context": "Total net sales 383,285 394,328 for twelve months ended September 30 2023 and September 24 2022",
            "expected": "383,285 million"
        },
        {
            "question": "What is Apple operating income?",
            "context": "Operating income 26,969 24,894 114,301 119,437 for three months and twelve months",
            "expected": "26,969 million"
        },
        {
            "question": "How much did Apple spend on research and development?",
            "context": "Research and development 7,307 6,761 29,915 26,251",
            "expected": "29,915 million annually"
        },
    ]

    def __init__(self):
        self.base_model_name = "google/flan-t5-base"
        self.finetuned_dir = Path("data/models/finetuned")
        log.info("ModelEvaluator initialized")

    def _load_base_model(self):
        """Load the original pre-trained model."""
        tokenizer = T5Tokenizer.from_pretrained(self.base_model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            self.base_model_name
        )
        model.eval()
        return tokenizer, model

    def _load_finetuned_model(self):
        """Load the LoRA fine-tuned model."""
        if not self.finetuned_dir.exists():
            log.error("Fine-tuned model not found. Run trainer first.")
            return None, None

        tokenizer = T5Tokenizer.from_pretrained(str(self.finetuned_dir))
        base_model = T5ForConditionalGeneration.from_pretrained(
            self.base_model_name
        )
        # Load LoRA weights on top of base model
        model = PeftModel.from_pretrained(
            base_model,
            str(self.finetuned_dir)
        )
        model.eval()
        return tokenizer, model

    def _generate_answer(
        self,
        tokenizer,
        model,
        question: str,
        context: str
    ) -> str:
        """Generate answer from model."""
        prompt = f"question: {question} context: {context}"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare_models(self):
        """
        Compare base model vs fine-tuned model side by side.

        This is how you prove fine-tuning actually helped.
        """
        log.info("Loading base model for comparison...")
        base_tokenizer, base_model = self._load_base_model()

        log.info("Loading fine-tuned model...")
        ft_tokenizer, ft_model = self._load_finetuned_model()

        print("\n" + "="*70)
        print("MODEL COMPARISON: Base vs Fine-Tuned")
        print("="*70)

        for i, test in enumerate(self.TEST_QUESTIONS, 1):
            question = test["question"]
            context = test["context"]
            expected = test["expected"]

            base_answer = self._generate_answer(
                base_tokenizer, base_model, question, context
            )

            print(f"\nTest {i}: {question}")
            print(f"Expected:    {expected}")
            print(f"Base model:  {base_answer}")

            if ft_model:
                ft_answer = self._generate_answer(
                    ft_tokenizer, ft_model, question, context
                )
                print(f"Fine-tuned:  {ft_answer}")

            print("-" * 50)