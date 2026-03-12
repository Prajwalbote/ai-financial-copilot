import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.load_index("apple_10k")

print("\n" + "="*60)
print("TEST 1 — Net Sales Question")
print("="*60)
result = rag.answer("What are Apple net sales?")
print(f"Answer: {result['answer']}")

print("\n" + "="*60)
print("TEST 2 — Operating Income")
print("="*60)
result = rag.answer("What is the operating income?")
print(f"Answer: {result['answer']}")

print("\n" + "="*60)
print("TEST 3 — Summarization")
print("="*60)
result = rag.summarize("net sales operating income shareholders equity")
print(f"Summary: {result['summary']}")