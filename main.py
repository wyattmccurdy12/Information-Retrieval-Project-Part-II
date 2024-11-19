'''
python main.py -q data/inputs/topics_1.json -d data/inputs/Answers.json -o data/outputs/
'''

import re
import string
from sentence_transformers import SentenceTransformer
import json
import argparse
import numpy as np
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
import os
import ast

class ResultRetrieverBM25:
    def __init__(self, queries_path, documents_path, outdir):
        self.queries_path = queries_path
        self.documents_path = documents_path
        self.outdir = outdir
        self.queries = None
        self.documents = None
        self.index = None

    def load_data(self):
        """
        Load queries and documents from JSON files into pandas DataFrames.
        """
        self.queries = pd.read_json(self.queries_path)
        self.documents = pd.read_json(self.documents_path)

    def preprocess_documents(self):
        """
        Preprocess documents to ensure they have the required fields for indexing.

        Clean the body field by removing html tags.
        """
        # Rename 'Id' column to 'docno' and stringify the 'docno' value
        self.documents.rename(columns={'Id': 'docno'}, inplace=True)
        self.documents['docno'] = self.documents['docno'].astype(str)

        # Apply clean_string_html
        self.documents['text'] = self.documents['Text'].apply(self.clean_string_html)

        # Remove punctuation
        self.documents['text'] = self.documents['text'].apply(self.remove_punctuation)

        # Drop 'Text' and 'Score' columns
        self.documents.drop(columns=['Text', 'Score'], inplace=True)

        # Save 100 rows of the dataframe to a tsv file
        self.documents.head(100).to_csv('sample_docs.tsv', sep='\t', index=False)

    def preprocess_queries(self):
        """
        Preprocess queries to ensure they have the required fields for retrieval.
    
        Clean the body field by removing html tags and punctuation.
        Clean the title field by removing punctuation.
        Clean the tags field by literally evaluating the stringified list and joining the tags.
        """
        # Preprocess title
        self.queries['Title'] = self.queries['Title'].apply(self.remove_punctuation)

        # Preprocess tags
        self.queries['Tags'] = self.queries['Tags'].apply(self.get_tags_str)

        # Rename 'Id' column to 'qid' and stringify the 'qid' value
        self.queries.rename(columns={'Id': 'qid'}, inplace=True)
        self.queries['qid'] = self.queries['qid'].astype(str)
    
        # Apply clean_string_html
        self.queries['query'] = self.queries['Body'].apply(self.clean_string_html)
    
        # Remove punctuation
        self.queries['query'] = self.queries['query'].apply(self.remove_punctuation)

        # Add title and tags to the query text
        self.queries['query'] = self.queries['Title'] + ' '  + self.queries['query'] + ' '  + self.queries['Tags']
    
        # Drop 'Body' column
        self.queries.drop(columns=['Body'], inplace=True)
    
        # Save 100 rows of the dataframe to a tsv file
        self.queries.head(100).to_csv('sample_queries.tsv', sep='\t', index=False)

    def remove_punctuation(self, text):
        """
        Replace punctuation in the given text with spaces.
        """
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translation_table)
    
    def build_index(self):
        """
        Initialize PyTerrier and build an index from the documents if it doesn't exist.
        Otherwise, load the existing index.

        This method checks if PyTerrier is started, initializes it if not,
        creates an indexer, and indexes the documents using the 'text' field.
        If the index directory already exists, it is loaded instead of being recreated.
        """
        if not pt.started():
            pt.init()
        
        index_dir = "./index"
        
        if os.path.exists(index_dir):
            print(f"Loading existing index from: {index_dir}")
            self.index = pt.IndexFactory.of(index_dir)
        else:
            print(f"Creating new index at: {index_dir}")
            os.makedirs(index_dir, exist_ok=True)
            print(f"Created index directory: {index_dir}")

            # Ensure the directory has read and write permissions
            os.chmod(index_dir, 0o755)
            print(f"Set read and write permissions for index directory: {index_dir}")
            
            indexer = pt.IterDictIndexer(index_dir, meta={'docno': 20, 'text': 4096}, stemmer='porter', stopwords='terrier')

            # Index the documents using the DataFrame directly
            self.index = indexer.index(self.documents.to_dict(orient='records'))

    def clean_string_html(self, html_str):
        '''
        Given a string with html tags, return a string without the html tags.
        '''
        clean_text = re.sub(r'<[^>]+>', '', html_str)
        return clean_text

    def get_tags_str(self, tags_str):
        '''
        Given an input string that represents a python list, 
        return a string with just the tags separated by spaces.
        '''
        tags_list = ast.literal_eval(tags_str)
        return ' '.join(tags_list)

class ResultRetrieverSentTrans:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, texts):
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
    Remove HTML tags from the input text using regular expressions.

    Parameters:
    text (str): The text from which to remove HTML tags.

    Returns:
    str: The text with HTML tags removed.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def main():
    """
    Main function to run the ranking and re-ranking system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ranking and Re-Ranking System')
    parser.add_argument('-q', '--queries', required=True, help='Path to the queries file')
    parser.add_argument('-d', '--documents', required=True, help='Path to the documents file')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory for experiment results')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    documents, queries = load_data(args.queries, args.documents)
    print("Data loaded successfully.")

    # Instantiate retrievers
    bm25_retriever = ResultRetrieverBM25(args.queries, args.documents, args.outdir)
    bm25_retriever.load_data()
    bm25_retriever.preprocess_documents()
    bm25_retriever.preprocess_queries()
    bm25_retriever.build_index()

    sent_trans_retriever = ResultRetrieverSentTrans()
    
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
        
        encoded_queries = sent_trans_retriever.encode([q[1] for q in processed_queries])
        
        processed_documents = {}
        for doc in tqdm(data_documents, desc="Processing documents"):
            doc_id = doc['Id']
            text = remove_html_tags(doc['Text'])
            processed_documents[doc_id] = text
        
        encoded_documents = sent_trans_retriever.encode(list(processed_documents.values()))
        
        return processed_queries, encoded_queries, processed_documents, encoded_documents

    # Process and encode queries and documents
    print("Processing and encoding queries and documents...")
    processed_queries, encoded_queries, processed_documents, encoded_documents = process_and_encode(queries, documents)

    # Function to perform initial ranking
    def perform_initial_ranking(processed_queries, encoded_queries, encoded_documents, processed_documents):
        initial_rankings = {}
        for query_id, query_text in tqdm(processed_queries, desc="Ranking queries"):
            query_embedding = sent_trans_retriever.encode([query_text])[0]
            scores = np.dot(encoded_documents, query_embedding)
            ranked_doc_indices = np.argsort(scores)[::-1][:100]
            initial_rankings[query_id] = [(list(processed_documents.keys())[doc_id], scores[doc_id]) for doc_id in ranked_doc_indices]
        return initial_rankings

    # Perform initial ranking for SentenceTransformer
    print("Performing initial ranking with SentenceTransformer...")
    initial_rankings_sent_trans = perform_initial_ranking(processed_queries, encoded_queries, encoded_documents, processed_documents)

    # Write initial rankings to TSV for SentenceTransformer
    output_filename_sent_trans = os.path.join(args.outdir, "results_bi.tsv")
    write_results_to_tsv(initial_rankings_sent_trans, output_filename_sent_trans, model_type='simple', model_status='pretrained')
    print(f"Initial rankings for SentenceTransformer have been computed and saved to {output_filename_sent_trans}.")

    # Perform initial ranking for BM25
    print("Performing initial ranking with BM25...")
    bm25 = pt.BatchRetrieve(bm25_retriever.index, wmodel='BM25')
    initial_rankings_bm25 = bm25.transform(bm25_retriever.queries)

    # Write initial rankings to TSV for BM25
    output_filename_bm25 = os.path.join(args.outdir, "results_bm.tsv")
    initial_rankings_bm25.to_csv(output_filename_bm25, sep='\t', index=False, header=False)
    print(f"Initial rankings for BM25 have been computed and saved to {output_filename_bm25}.")

if __name__ == "__main__":
    main()