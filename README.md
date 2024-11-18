# Information Retrieval System for AskUbuntu QA
## Project Description

This project is meant to develop an information retrieval system using data from the classic AskUbuntu website. 
Given a user question, we would like to reliably pull up results which satisfy the user's information needs.  
In this section of the project - part 2 - we compare more traditional models with sentence transformer models. 
In previous work, we developed a bm-25 system which worked reasonably well. 
We will compare the BM-25 system with a pre-trained sentence transformer and measure whether there are significant differences
between the two approaches on metrics including:
* nDCG@K
* p@k
* mAP
* bpref

## Models Used

- BM25 
- `sentence-transformers/all-MiniLM-L6-v2`

## Requirements
- Python 3.6+
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `beautifulsoup4`
- `tqdm`
- `numpy`

You can install the required packages using pip:

```bash
pip install transformers sentence-transformers scikit-learn beautifulsoup4 tqdm numpy
```


## Sample command line command to get retrieval results.

```bash
export TRANSFORMERS_CACHE=/transformer_cache/
python main.py -q data/inputs/topics_1.json -d data/inputs/Answers.json
```