from sklearn.neighbors import NearestNeighbors
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM,  AutoTokenizer

def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    return documents

def create_embeddings(documents, model):
    embeddings = model.encode(documents, convert_to_tensor=True)
    return np.array(embeddings)

def build_index(embeddings):
    index = NearestNeighbors(n_neighbors=5, metric='euclidean')  # or 'cosine'
    index.fit(embeddings)
    return index

def retrieve_documents(query, model, index, documents, k=1):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.kneighbors(query_embedding, n_neighbors=k)
    print(f"Distances: {distances}, Indices: {indices}")
    return [documents[idx] for idx in indices[0]]

def generate_answer_stream(query, retrieved_docs, tokenizer, model):
    """
    Generate answer using streaming tokens (generator).
    Only works with models that support `generate` with `return_dict_in_generate=True` and `output_scores=True`.
    """
    import torch

    context = "\n".join(retrieved_docs)
    prompt = f"Context: {context}\nAnswer with facts from the context. Here is the question: {query}\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Stream tokens one by one
    streamer = None
    try:
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer)
    except ImportError:
        pass  # Fallback to manual decoding if TextStreamer is not available

    # Generate response with streaming
    output_ids = []
    for output in model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        num_return_sequences=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    ).sequences[0]:
        output_ids.append(output.item())
        if streamer:
            streamer.put(torch.tensor([output]))
        else:
            yield tokenizer.decode([output], skip_special_tokens=True)

    if streamer:
        streamer.end()

def main():
    # Load data
    print("Loading data...")
    data_file = "./data.txt"
    documents = load_data_from_file(data_file)
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        print("No documents found in the file.")
        return
    
    print("Loading retriever model...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings and index
    embeddings = create_embeddings(documents, retriever_model)
    index = build_index(embeddings)    
    #print (embeddings.shape)    
    print("Index built successfully.")
    
    model_name = "google/flan-t5-base"
    
    print(f"Loading generator model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #gen_model = AutoModelForCausalLM.from_pretrained(model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Generator model loaded successfully.")

    # Example query
    query = "What materials should I use to build a playhouse outside for my children?"
    

    print(f"Query: {query}")
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, retriever_model, index, documents, k=1)
    print(f"Retrieved document: {retrieved_docs}")

    #answer = generate_answer_stdalone(query, tokenizer, gen_model)
    #answer = generate_answer(query, retrieved_docs, tokenizer, gen_model)
    print("Streaming Answer:", end=" ", flush=True)
    for token in generate_answer_stream(query, retrieved_docs, tokenizer, gen_model):
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    main()