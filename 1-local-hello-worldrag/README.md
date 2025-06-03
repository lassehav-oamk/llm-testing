# RAG Hello World

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using Python.

## Files

- `rag-hello-world.py`: Main script implementing the RAG example.
- `data-txt`: Examples of custom data, which the LLM will be using in the example. 

## Requirements

- Python 3.8+
  
To create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Then install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python rag-hello-world.py
```

## Description

The script retrieves relevant information from a local knowledge base and generates a response using a language model.

## How It Works

Let's break down what happens in `rag-hello-world.py` step by step:

1. **Load the Knowledge Base**  
   The script reads custom data from the `data-txt` file(s). This data acts as the knowledge base that the language model will use to answer questions.

2. **Create embeddings of the loaded data**  
   The all-MiniLM-L6-v2 pretrained transformer model is from Hugging Face and is designed to create embeddings of short texts, eg. vector representations of them. It is fast and comes with 256 token input limit. 

3. **Retrieve Relevant Information**  
   Using simple keyword matching or embedding-based search (depending on the implementation), the script finds the most relevant pieces of information from the knowledge base that relate to the user's query. 

4. **Generate a Response**  
   The script passes the retrieved information along with the original query to a local language model Flan-T5 from Google. The model uses both the query and the retrieved context to generate a more accurate and informed answer. The query in the example is hardcoded. Model FlanT5-base is used in this example, it is finetuned for answering questions (https://arxiv.org/pdf/2210.11416v5). 
   

5. **Display the Answer**  
   Finally, the script prints the generated answer for the user to see.

**Why use RAG?**  
Retrieval-Augmented Generation combines the strengths of search (retrieval) and language models (generation). It helps the model provide up-to-date and contextually relevant answers, even if the model itself wasn't trained on the latest data.


