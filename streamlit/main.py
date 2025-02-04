import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing_extensions import Union, List, Tuple, Optional, Dict

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline,
    T5ForQuestionAnswering,
    BertForQuestionAnswering
)
import torch
import time
from dataclasses import dataclass
import json

def prepare_and_insert_data(
    paragraphs: Union[np.ndarray, List[str]], 
    collection: chromadb.Collection,
    batch_size: int = 500
) -> None:
    """
    Prepare and insert data into ChromaDB with proper handling of NumPy arrays
    
    Args:
        paragraphs: Array or list of text paragraphs
        collection: ChromaDB collection instance
        batch_size: Number of items to process in each batch
    """
    # Convert NumPy array to list if necessary
    if isinstance(paragraphs, np.ndarray):
        paragraphs = paragraphs.tolist()
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings
    embeddings = model.encode(paragraphs).tolist()
    
    # Create document IDs and metadata
    doc_ids = [f"doc_{i}" for i in range(len(paragraphs))]
    metadatas = [{"index": i} for i in range(len(paragraphs))]
    
    # Verify lengths
    print(f"Verification counts:")
    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"IDs: {len(doc_ids)}")
    print(f"Metadata: {len(metadatas)}")
    
    # Add data to ChromaDB in batches
    for i in range(0, len(paragraphs), batch_size):
        batch_end = min(i + batch_size, len(paragraphs))
        collection.add(
            documents=paragraphs[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=doc_ids[i:batch_end]
        )
        if (i + batch_size) % 5000 == 0:
            print(f"Processed {i + batch_size} entries...")
    
    print(f"Successfully added {len(paragraphs)} paragraphs to ChromaDB!")

def search_query(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 3,
    threshold: float = 0.01,
    return_raw: bool = False
) -> Union[None, List[Tuple[str, float]]]:
    """
    Search for relevant documents in ChromaDB collection.
    
    Args:
        query: Search query string
        collection: ChromaDB collection to search in
        model: SentenceTransformer model instance
        top_k: Number of results to return
        threshold: Minimum similarity score (0-1) to include in results
        return_raw: If True, returns raw results instead of printing
        
    Returns:
        If return_raw is True, returns list of (document, score) tuples
        If return_raw is False, prints results and returns None
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query]).tolist()
        
        # Perform search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        
        # Process results
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadata'][0] if 'metadata' in results else [{}] * len(documents)
        
        # Convert distance to similarity score (1 - distance)
        scores = [1 - dist for dist in distances]
        
        # Filter results by threshold
        filtered_results = [
            (doc, score, meta) 
            for doc, score, meta in zip(documents, scores, metadatas)
            if score >= threshold
        ]
        
        if return_raw:
            return filtered_results
        
        # Display results
        if not filtered_results:
            print("\nNo results found matching your query.")
            return None
            
        print(f"\nTop {len(filtered_results)} relevant paragraphs for query: '{query}'")
        print("-" * 80)
        
        for i, (doc, score, meta) in enumerate(filtered_results, 1):
            print(f"\n{i}. Similarity Score: {score:.4f}")
            if meta:
                print(f"Metadata: {meta}")
            print(f"Paragraph: {doc}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error performing search: {str(e)}")
        if return_raw:
            return []
    
    return None

@dataclass
class QAResult:
    """Store QA model results with metadata"""
    answer: str
    confidence: float
    context: str
    model_name: str
    processing_time: float

class QASystem:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.models = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize QA model on CPU to avoid memory issues"""
        try:
            print("Loading RoBERTa model on CPU...")
            self.models['roberta'] = pipeline(
                'question-answering',
                model='deepset/roberta-base-squad2',
                device=-1  # Force CPU usage
            )
            print("Successfully loaded RoBERTa model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def process_search_results(
        self,
        results: List[Tuple[str, float, Dict]],
        min_score: float = 0.01
    ) -> str:
        """Process and combine search results into context."""
        # Sort results by descending score
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Filter by minimum score
        relevant_docs = [doc for doc, score, _ in results if score >= min_score]

        # Return concatenated documents or None if no relevant docs
        return " ".join(relevant_docs) if relevant_docs else ""


    def answer_question(
        self,
        question: str,
        context: str,
        model_name: str = 'roberta'
    ) -> Optional[QAResult]:
        """Generate answer using the model"""
        if not self.models:
            print("No models available.")
            return None
            
        start_time = time.time()
        
        try:
            result = self.models[model_name](
                question=question,
                context=context
            )
            
            processing_time = time.time() - start_time
            
            return QAResult(
                answer=result['answer'],
                confidence=result['score'],
                context=context,
                model_name=model_name,
                processing_time=processing_time
            )
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return None
        
    def rephrase_answer(self, question: str, answer: str) -> str:
        """Rephrase the answer into a complete sentence."""
        if "year" in question.lower():
            return f"The year when {question[3:].strip().lower()} is {answer}."
        elif "how" in question.lower():
            return f"The way {question[4:].strip().lower()} happened is because {answer}."
        elif "why" in question.lower():
            return f"The reason {question[4:].strip().lower()} is that {answer}."
        else:
            return f"The answer to '{question}' is: {answer}."


def get_answer(
    question: str,
    search_results: List[Tuple[str, float, Dict]],
    collection,
    embedding_model,
    min_score: float = 0.01
) -> Optional[str]:
    """Complete pipeline to get answer from search results"""
    # Initialize QA system
    qa_system = QASystem(collection, embedding_model)
    
    # Process search results into context
    context = qa_system.process_search_results(search_results, min_score)
    if not context:
        print("No relevant context found with sufficient confidence score.")
        return None
    
    # Get answer
    result = qa_system.answer_question(question, context)
    
    if result:
        if result.confidence < 0.01:
            print("Confidence score is below 0.02, consider verifying the answer.")
        print("\nQuestion:", question)
        print("\nAnswer:", result.answer)
        print("Confidence:", f"{result.confidence:.4f}")
        print("Processing Time:", f"{result.processing_time:.2f}s")
        print("\nRelevant Context Used:")
        print("-" * 80)
        print(context)
        return result.answer
    return None

def ingest(csv_path):
    df = pd.read_csv(csv_path)
    paragraphs = df["Text"].to_list()

    # Initialize ChromaDB client with the new architecture
    client = chromadb.PersistentClient(
        path="./chroma_storage"  # Update this path as needed
    )

    # Create a collection (if it doesn't exist)
    collection_name = "paragraph_collection"
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)

    client = chromadb.PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection("paragraph_collection")
    prepare_and_insert_data(paragraphs, collection)

if __name__ == "__main__":

    df = pd.read_csv('../data/train_data.csv')
    df.head()

    grouped = df.groupby('Paragraph')

    json_list = []
    for paragraph, group in grouped:
        questions = []
        for _, row in group.iterrows():
            questions.append({
                "q": row['Question'],
                "a": row['Answer_text'],
                "a_start": row['Answer_start']
            })
        json_list.append({
            "paragraph": paragraph,
            "questions": questions
        })

    paragraphs = []

    with open('context.txt', 'w', encoding='utf-8') as file:
        for paragraph in json_list:
            # text = paragraph["paragraph"] + '\n' + '\n'.join([f'{q["q"]} - {q["a"]}' for q in paragraph["questions"]]) + '\n'
            text = paragraph["paragraph"]
            paragraphs.append(text)
            file.write(text)

    # Initialize ChromaDB client with the new architecture
    client = chromadb.PersistentClient(
        path="./chroma_storage"  # Update this path as needed
    )

    # Create a collection (if it doesn't exist)
    collection_name = "paragraph_collection"
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)

    client = chromadb.PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection("paragraph_collection")
    prepare_and_insert_data(paragraphs, collection)
