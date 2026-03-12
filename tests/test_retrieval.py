import sys
import os

# Add project root to Python path
# This tells Python where to find our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.batch_embedder import BatchEmbedder
from retrieval.retriever import Retriever

# Step 1 - Load embeddings we created in Phase 3
embedder = BatchEmbedder()
embeddings, chunks = embedder.load_embeddings('apple_10k')

# Step 2 - Build and save the FAISS index
retriever = Retriever()
retriever.build_and_save(embeddings, chunks, 'apple_10k')

# Step 3 - Ask financial questions!
questions = [
    'What is Apple total revenue?',
    'What are the operating expenses?',
    'How much did Apple spend on research?',
]

for question in questions:
    print(f'\nQ: {question}')
    results = retriever.retrieve(question, top_k=2)
    for r in results:
        score = r['score']
        text = r['text'][:150]
        print(f'  Score: {score:.3f} | {text}')