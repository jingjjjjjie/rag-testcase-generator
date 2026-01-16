"""
Test script for GenericPreprocessor with simulated document content.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.generic_preprocessor import GenericPreprocessor


# Simulated document content - a technical article about machine learning
SAMPLE_DOCUMENT = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

The process begins with observations or data, such as examples, direct experience, or instruction. The goal is to allow computers to learn automatically without human intervention and adjust actions accordingly.

Types of Machine Learning

There are three main types of machine learning algorithms:

Supervised Learning
In supervised learning, the algorithm learns from labeled training data. The training data includes both input features and the correct output labels. Common applications include spam detection, image classification, and price prediction. The algorithm makes predictions based on the patterns it discovers in the training data.

Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or structures in the input data without any guidance. Clustering and dimensionality reduction are common unsupervised learning techniques. Examples include customer segmentation and anomaly detection.

Reinforcement Learning
Reinforcement learning is about taking suitable actions to maximize reward in a particular situation. The agent learns to achieve a goal in an uncertain environment. It employs trial and error to come up with a solution to the problem. The agent gets either rewards or penalties for the actions it performs.

Applications of Machine Learning

Machine learning has numerous real-world applications across various industries:

Healthcare: Disease diagnosis, drug discovery, and personalized treatment recommendations.

Finance: Fraud detection, algorithmic trading, and credit scoring.

Transportation: Self-driving cars, route optimization, and predictive maintenance.

E-commerce: Product recommendations, dynamic pricing, and customer churn prediction.

Challenges and Considerations

Despite its power, machine learning faces several challenges. Data quality is crucial because models are only as good as the data they are trained on. Overfitting occurs when a model learns the training data too well and fails to generalize to new data. Interpretability is another concern, especially in sensitive domains like healthcare and finance where understanding the reasoning behind predictions is important.

Conclusion

Machine learning continues to evolve rapidly, with new techniques and applications emerging regularly. Understanding the fundamentals is essential for anyone looking to leverage this powerful technology. As computational power increases and more data becomes available, the potential applications of machine learning will only continue to grow.
"""


def test_chunking_only():
    """Test just the chunking functionality without LLM calls."""
    print("=" * 80)
    print("Testing GenericPreprocessor - Chunking Only")
    print("=" * 80)

    # Create preprocessor with smaller chunks for demonstration
    preprocessor = GenericPreprocessor(
        chunk_size=500,
        chunk_overlap=50
    )

    # Test the chunking directly
    chunks = preprocessor.chunk_text(SAMPLE_DOCUMENT)

    print(f"\nOriginal document length: {len(SAMPLE_DOCUMENT)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Chunk size setting: {preprocessor.chunk_size}")
    print(f"Chunk overlap setting: {preprocessor.chunk_overlap}")

    print("\n" + "-" * 80)
    print("Chunks Preview:")
    print("-" * 80)

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i + 1}] ({len(chunk)} chars)")
        print(f"Preview: {chunk[:100]}...")

    return chunks


def test_sentence_labeling():
    """Test the sentence labeling functionality."""
    print("\n" + "=" * 80)
    print("Testing Sentence Labeling")
    print("=" * 80)

    preprocessor = GenericPreprocessor()

    sample_text = """Machine learning is a powerful technology. It can solve complex problems.

Deep learning is a subset of machine learning. Neural networks are the foundation of deep learning."""

    labeled_text, final_counter = preprocessor.add_sentence_labels(sample_text)

    print(f"\nOriginal text:\n{sample_text}")
    print(f"\nLabeled text:\n{labeled_text}")
    print(f"\nTotal sentences labeled: {final_counter - 1}")

    return labeled_text


def test_full_process():
    """Test the full processing pipeline (without PDF, using text directly)."""
    print("\n" + "=" * 80)
    print("Testing Full Processing Pipeline")
    print("=" * 80)

    preprocessor = GenericPreprocessor(
        chunk_size=600,
        chunk_overlap=80
    )

    # Process chunks manually (simulating what run() does internally)
    chunks = preprocessor.chunk_text(SAMPLE_DOCUMENT)
    chunk_contents = preprocessor.process_chunks(chunks, doc_id="ml_article")

    print(f"\nProcessed {len(chunk_contents)} chunks")

    print("\n" + "-" * 80)
    print("Processed Chunks Detail:")
    print("-" * 80)

    for chunk in chunk_contents:
        print(f"\n[{chunk['id']}] - {chunk['char_count']} chars")
        print(f"Original: {chunk['origin_context'][:80]}...")
        print(f"Labeled:  {chunk['context'][:80]}...")

    return chunk_contents


def test_different_chunk_sizes():
    """Compare different chunk size configurations."""
    print("\n" + "=" * 80)
    print("Comparing Different Chunk Sizes")
    print("=" * 80)

    configs = [
        {"chunk_size": 300, "chunk_overlap": 30},
        {"chunk_size": 500, "chunk_overlap": 50},
        {"chunk_size": 800, "chunk_overlap": 100},
        {"chunk_size": 1200, "chunk_overlap": 150},
    ]

    print(f"\nDocument length: {len(SAMPLE_DOCUMENT)} chars\n")
    print(f"{'Chunk Size':<12} {'Overlap':<10} {'Num Chunks':<12} {'Avg Size':<10}")
    print("-" * 50)

    for config in configs:
        preprocessor = GenericPreprocessor(**config)
        chunks = preprocessor.chunk_text(SAMPLE_DOCUMENT)
        avg_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

        print(f"{config['chunk_size']:<12} {config['chunk_overlap']:<10} {len(chunks):<12} {avg_size:<10.1f}")


if __name__ == "__main__":
    # Run all tests
    test_chunking_only()
    test_sentence_labeling()
    test_full_process()
    test_different_chunk_sizes()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
