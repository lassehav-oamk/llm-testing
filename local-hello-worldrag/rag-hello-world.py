import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM,  AutoTokenizer
import torch

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    return documents

def create_embeddings(documents, model):
    embeddings = model.encode(documents, convert_to_tensor=True)
    return np.array(embeddings)

def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_documents(query, model, index, documents, k=1):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[idx] for idx in indices[0]]

def generate_answer(query, retrieved_docs, tokenizer, model):
    # Combine query and retrieved documents into a prompt
    context = "\n".join(retrieved_docs)
    #prompt = f"Context: {context}. \n\nUse only the provided context to answer the following question with a single number, word, or short phrase: {query} \nAnswer: "
    prompt = f"Context: {context}\nAnswer with facts from the context: {query}\nAnswer: "


    # Tokenize input
    #inputs = tokenizer(prompt, return_tensors="pt")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)


    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,        
        num_return_sequences=1,
        no_repeat_ngram_size=0, #Prevents bigram repetition, which can force gpt2 to diversify its output unnaturally, leading to incoherent text like the population statistics.
        do_sample=False, # Random sampling increases the likelihood of gpt2 generating hallucinated or off-topic text, even with relevant context.
        top_k=10,
        top_p=0.9,
        temperature=0.7, # Lower temperature for more deterministic output
    )

    # Decode and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #return answer.split("Answer: ")[-1].strip()
    return answer

def generate_answer_stdalone(query, tokenizer, model):
    # Combine query and retrieved documents into a prompt    
    #prompt = f"Question: {query}\n: Answer:"
    #prompt = f"Complete the following Python code, output only code which is required to complete ongoing line or next lines to finish ongoing function:\n```python\n{query}\n```"
    #prompt = f"python\n{query}\n"
    prompt = query

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=100,
        max_new_tokens=70,
        num_return_sequences=1,
        no_repeat_ngram_size=2, #Prevents bigram repetition, which can force gpt2 to diversify its output unnaturally, leading to incoherent text like the population statistics.
        do_sample=True, # Random sampling increases the likelihood of gpt2 generating hallucinated or off-topic text, even with relevant context.
        top_k=80,
        top_p=0.9,
        temperature=0.7      
    )

    # Decode and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #return answer.split("```python\n")[-1].strip()
    return answer

def main():
    # # Load data
    data_file = "./data.txt"
    documents = load_data(data_file)
    
    # Initialize retriever (sentence-transformers)
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings and FAISS index
    embeddings = create_embeddings(documents, retriever_model)
    index = build_index(embeddings)    
    #print (embeddings.shape)    
    
    # Initialize generative model (use a small model for simplicity)
    #model_name = "distilgpt2"  # Small model for quick testing
    #model_name = "openai-community/gpt2"
    #model_name = "facebook/bart-large"
    #model_name  = "microsoft/CodeGPT-small-py"
    #model_name = "Salesforce/codegen-350M-mono"  # Small model for quick testing
    model_name = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #gen_model = AutoModelForCausalLM.from_pretrained(model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Example query
    #query = "How many countries are there in the world?"
    query = "What is good material for building a construction to my backyard?"

    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, retriever_model, index, documents, k=1)
    print(f"Retrieved document: {retrieved_docs[0]}")

    #answer = generate_answer_stdalone(query, tokenizer, gen_model)
    answer = generate_answer(query, retrieved_docs, tokenizer, gen_model)
    print("Standalone Answer:", answer)

if __name__ == "__main__":
    main()