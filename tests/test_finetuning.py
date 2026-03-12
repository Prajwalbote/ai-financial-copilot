import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetuning.dataset_prep import DatasetPreparator
from finetuning.trainer import FinancialModelTrainer
from finetuning.evaluate import ModelEvaluator

print("="*60)
print("STEP 1 — Preparing Dataset")
print("="*60)
preparator = DatasetPreparator()
train_dataset, val_dataset = preparator.prepare_dataset()
print(f"Train examples: {len(train_dataset)}")
print(f"Val examples:   {len(val_dataset)}")

print("\n" + "="*60)
print("STEP 2 — Fine-Tuning Model")
print("="*60)
print("This takes 5-15 minutes on CPU...")
trainer = FinancialModelTrainer()
trainer.train(train_dataset, val_dataset)

print("\n" + "="*60)
print("STEP 3 — Evaluating Results")
print("="*60)
evaluator = ModelEvaluator()
evaluator.compare_models()