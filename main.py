'''
Wyatt McCurdy
Information Retrieval Project Part II
November 18, 2024
Dr. Behrooz Mansouri

In part II of this project, we move towards the use of sentence transformers 
to rank inputs. This is expected to outperform models that were used
in assignment part 1. We will use a bi-encoder, and skip many steps that we
used in part 1. We will abstain from stopword removal and tokenization
of inputs, and trust the sentence transformer to reliably encode input
queries and documents. 

The query-document pairs will be split into train/test/validation sets
with a 80/10/10 split.

Data in the data directory is in trec format. 
documents: data/inputs/Answers.json
queries: data/inputs/topics_1.json, topics_2.json
qrels:   data/inputs/qrel_1.tsv (qrel_2.tsv is reserved by the instructor)

models used: 
sentence-transformers/all-MiniLM-L6-v2
'''

import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import json
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from tqdm import tqdm

class Retriever:
    """
    A class to represent a retriever model which can be either a bi-encoder or a cross-encoder.

    Attributes:
    model_type (str): The type of the model ('bi-encoder' or 'cross-encoder').
    model (SentenceTransformer or CrossEncoder): The loaded model.
    """
    def __init__(self, model_type, model_name):
        """
        Initializes the Retriever with the specified model type and name.

        Parameters:
        model_type (str): The type of the model ('bi-encoder' or 'cross-encoder').
        model_name (str): The name of the model to load.
        """
        self.model_type = model_type
        if model_type == 'bi-encoder':
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError("Invalid model type. Choose 'bi-encoder'.")

    def encode(self, texts):
        """
        Encodes a list of texts using the bi-encoder model.

        Parameters:
        texts (list of str): The texts to encode.

        Returns:
        list: The encoded texts.
        """
        if self.model_type != 'bi-encoder':
            raise ValueError("Encode method is only available for bi-encoder.")
        return self.model.encode(texts)

def load_data(queries_path, documents_path):
    """
    Loads the documents and queries from the specified files.

    Parameters:
    queries_path (str): The path to the queries file.
    documents_path (str): The path to the documents file.

    Returns:
    tuple: A tuple containing the documents and queries.
    """
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    return documents, queries

def split_data(queries, documents, test_size=0.2, val_size=0.1):
    """
    Splits the data into train, validation, and test sets.

    Parameters:
    queries (list): The list of queries.
    documents (list): The list of documents.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the dataset to include in the validation split.

    Returns:
    tuple: A tuple containing the train, validation, and test sets.
    """
    # First split into train+val and test
    queries_train_val, queries_test, documents_train_val, documents_test = train_test_split(
        queries, documents, test_size=test_size, random_state=42)
    
    # Then split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size to account for the first split
    queries_train, queries_val, documents_train, documents_val = train_test_split(
        queries_train_val, documents_train_val, test_size=val_size_adjusted, random_state=42)
    
    return (queries_train, documents_train), (queries_val, documents_val), (queries_test, documents_test)

def write_results_to_tsv(results, output_filename, model_type, model_status):
    """
    Writes the results to a TSV file in TREC format.

    Parameters:
    results (dict): The results to write.
    output_filename (str): The name of the output file.
    model_type (str): The type of the model ('simple' or 'reranked').
    model_status (str): The status of the model ('pretrained' or 'finetuned').
    """
    with open(output_filename, 'w') as f:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_type}_{model_status}\n")

def remove_html_tags(text):
    """
    Remove HTML tags from the input text using BeautifulSoup.

    Parameters:
    text (str): The text from which to remove HTML tags.

    Returns:
    str: The text with HTML tags removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\\", "", text)
    return text

def main():
    """
    Main function to run the ranking and re-ranking system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ranking and Re-Ranking System')
    parser.add_argument('-q', '--queries', required=True, help='Path to the queries file')
    parser.add_argument('-d', '--documents', required=True, help='Path to the documents file')
    parser.add_argument('-be', '--bi_encoder', required=True, help='Bi-encoder model string')
    parser.add_argument('-ft', '--finetuned', action='store_true', help='Indicate if the model is fine-tuned')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    documents, queries = load_data(args.queries, args.documents)
    print("Data loaded successfully.")

    # Split data into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    (train_queries, train_documents), (val_queries, val_documents), (test_queries, test_documents) = split_data(queries, documents)
    print("Data split successfully.")

    # Instantiate retriever
    bi_encoder_retriever = Retriever('bi-encoder', args.bi_encoder)
    
    # Function to process and encode queries and documents
    def process_and_encode(data_queries, data_documents):
        processed_queries = []
        for query in tqdm(data_queries, desc="Processing queries"):
            query_id = query['Id']
            title = remove_html_tags(query['Title'])
            body = remove_html_tags(query['Body'])
            tags = ' '.join(query['Tags'])
            merged_query = f"{title} {body} {tags}"
            processed_queries.append((query_id, merged_query))
        
        encoded_queries = bi_encoder_retriever.encode([q[1] for q in processed_queries])
        
        processed_documents = {}
        for doc in tqdm(data_documents, desc="Processing documents"):
            doc_id = doc['Id']
            text = remove_html_tags(doc['Text'])
            processed_documents[doc_id] = text
        
        encoded_documents = bi_encoder_retriever.encode(list(processed_documents.values()))
        
        return processed_queries, encoded_queries, processed_documents, encoded_documents

    # Process and encode train, validation, and test sets
    print("Processing and encoding train set...")
    train_processed_queries, train_encoded_queries, train_processed_documents, train_encoded_documents = process_and_encode(train_queries, train_documents)
    print("Processing and encoding validation set...")
    val_processed_queries, val_encoded_queries, val_processed_documents, val_encoded_documents = process_and_encode(val_queries, val_documents)
    print("Processing and encoding test set...")
    test_processed_queries, test_encoded_queries, test_processed_documents, test_encoded_documents = process_and_encode(test_queries, test_documents)

    # Function to perform initial ranking
    def perform_initial_ranking(processed_queries, encoded_queries, encoded_documents, processed_documents):
        initial_rankings = {}
        for query_id, query_text in tqdm(processed_queries, desc="Ranking queries"):
            query_embedding = bi_encoder_retriever.encode([query_text])[0]
            scores = np.dot(encoded_documents, query_embedding)
            ranked_doc_indices = np.argsort(scores)[::-1][:100]
            initial_rankings[query_id] = [(list(processed_documents.keys())[doc_id], scores[doc_id]) for doc_id in ranked_doc_indices]
        return initial_rankings

    # Perform initial ranking for validation and test sets
    print("Performing initial ranking for validation set...")
    val_initial_rankings = perform_initial_ranking(val_processed_queries, val_encoded_queries, val_encoded_documents, val_processed_documents)
    print("Performing initial ranking for test set...")
    test_initial_rankings = perform_initial_ranking(test_processed_queries, test_encoded_queries, test_encoded_documents, test_processed_documents)

    # Extract topic number from queries file name
    topic_number = args.queries.split('_')[-1].split('.')[0]

    # Determine output filenames
    val_output_filename = f"result_bi_val{'_ft' if args.finetuned else ''}_{topic_number}.tsv"
    test_output_filename = f"result_bi_test{'_ft' if args.finetuned else ''}_{topic_number}.tsv"

    # Write initial rankings to TSV
    write_results_to_tsv(val_initial_rankings, val_output_filename, model_type='simple', model_status='finetuned' if args.finetuned else 'pretrained')
    write_results_to_tsv(test_initial_rankings, test_output_filename, model_type='simple', model_status='finetuned' if args.finetuned else 'pretrained')
    print(f"Initial rankings have been computed and saved to {val_output_filename} and {test_output_filename}.")

if __name__ == "__main__":
    main()