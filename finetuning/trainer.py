"""
finetuning/trainer.py — LoRA Fine-Tuning
=========================================
WHY LORA?
    Full fine-tuning: update all 250M parameters
    LoRA fine-tuning: update only ~1M parameters

    LoRA adds small trainable matrices to attention layers.
    These matrices learn the domain-specific adjustments
    while the original weights stay frozen.

    Result: 99% less memory, 10x faster, similar quality.

WHAT YOU LEARN:
    - How LoRA works conceptually
    - HuggingFace Trainer API
    - Training loop best practices
    - Model saving and loading
"""

import torch
from pathlib import Path
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class FinancialModelTrainer:
    """
    Fine-tunes Flan-T5 on financial QA using LoRA.

    Usage:
        trainer = FinancialModelTrainer()
        trainer.train(train_dataset, val_dataset)
        # Model saved to data/models/finetuned/
    """

    def __init__(self):
        self.cfg = get_config()
        self.model_name = self.cfg.finetuning.base_model
        self.output_dir = Path(self.cfg.finetuning.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None

        log.info(
            f"FinancialModelTrainer initialized | "
            f"base_model={self.model_name}"
        )

    def _load_base_model(self):
        """Load base model and apply LoRA adapters."""
        log.info(f"Loading base model: {self.model_name}")

        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        base_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )

        # ── Apply LoRA ──────────────────────────────────────────
        # WHY THESE SPECIFIC TARGET MODULES?
        # q = query, v = value matrices in attention layers
        # These are the most impactful for domain adaptation
        # Adding more modules improves quality but uses more memory
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.cfg.finetuning.lora_r,
            lora_alpha=self.cfg.finetuning.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],
        )

        self._model = get_peft_model(base_model, lora_config)

        # Print trainable parameter count
        trainable, total = self._count_parameters()
        log.info(
            f"LoRA applied | "
            f"trainable={trainable:,} | "
            f"total={total:,} | "
            f"percentage={100*trainable/total:.2f}%"
        )

    def _count_parameters(self) -> tuple[int, int]:
        """Count trainable vs total parameters."""
        trainable = sum(
            p.numel() for p in self._model.parameters()
            if p.requires_grad
        )
        total = sum(
            p.numel() for p in self._model.parameters()
        )
        return trainable, total

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize dataset for training.

        Converts text to token IDs that the model understands.
        Labels are the target token IDs.
        -100 in labels means ignore this token in loss calculation.
        """
        max_input_length = 512
        max_target_length = 128

        def tokenize_fn(examples):
            inputs = self._tokenizer(
                examples["input_text"],
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
            )

            targets = self._tokenizer(
                examples["target_text"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )

            # Replace padding token id with -100
            # so loss is not computed on padding tokens
            labels = [
                [(t if t != self._tokenizer.pad_token_id else -100)
                 for t in target]
                for target in targets["input_ids"]
            ]

            inputs["labels"] = labels
            return inputs

        return dataset.map(tokenize_fn, batched=True)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        """
        Run the fine-tuning training loop.

        Args:
            train_dataset: Training examples
            val_dataset:   Validation examples

        Training loop:
            For each epoch:
                For each batch:
                    Forward pass → compute loss
                    Backward pass → compute gradients
                    Update ONLY LoRA parameters
                Evaluate on validation set
                Save checkpoint if improved
        """
        self._load_base_model()

        log.info("Tokenizing datasets...")
        train_tokenized = self._tokenize_dataset(train_dataset)
        val_tokenized = self._tokenize_dataset(val_dataset)

        # ── Training Arguments ──────────────────────────────────
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.cfg.finetuning.num_epochs,
            per_device_train_batch_size=self.cfg.finetuning.batch_size,
            per_device_eval_batch_size=self.cfg.finetuning.batch_size,
            learning_rate=self.cfg.finetuning.learning_rate,
            warmup_steps=10,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=5,
            report_to="none",
            use_cpu=True,           # Force CPU (set False if you have GPU)
        )

        # Data collator handles batching and padding
        data_collator = DataCollatorForSeq2Seq(
            self._tokenizer,
            model=self._model,
            padding=True,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
        )

        log.info("Starting fine-tuning...")
        log.info(
            f"epochs={self.cfg.finetuning.num_epochs} | "
            f"batch_size={self.cfg.finetuning.batch_size} | "
            f"lr={self.cfg.finetuning.learning_rate}"
        )

        trainer.train()

        # Save final model
        self._model.save_pretrained(str(self.output_dir))
        self._tokenizer.save_pretrained(str(self.output_dir))

        log.info(
            f"Fine-tuning complete | "
            f"model saved to {self.output_dir}"
        )
        return trainer