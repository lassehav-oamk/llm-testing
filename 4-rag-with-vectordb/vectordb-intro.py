from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
import google.generativeai as genai
from sailing_documents import exampleSourceDocuments


# Initialize environment variable for Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize sentence transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')



# Query Gemini API
def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

# Retrieve context from vector database
def create_context(question):
    results = queryVectorDb(question)
    # Extract text from search results to create context
    context_texts = []
    for result in results[0]:
        context_texts.append(result.entity.text)
    return " ".join(context_texts)

# Main RAG function
def rag_query(question):
    # Retrieve context
    context = create_context(question)
    # Create prompt for Gemini
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    # Get response from Gemini
    answer = query_gemini(prompt)
    return answer

def initVectorDb():
    print("Initializing vector database...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(exampleSourceDocuments)

    connections.connect("default", host="localhost", port="19530")

    collection_name = "sailing_knowledge_base"
    
    # Check if collection already exists
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        collection = Collection(name=collection_name)
        collection.load()
        
        # Check if collection has data
        num_entities = collection.num_entities
        if num_entities > 0:
            print(f"Collection already contains {num_entities} documents. Clearing existing data...")
            collection.delete(expr="id >= 0")
            collection.flush()
            print("Existing data cleared.")
        else:
            print("Collection exists but is empty.")
    else:
        print("Creating new collection...")
        # Define the fields in our database collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]

        # Create a schema by using the fields defined above
        schema = CollectionSchema(fields, description="Sailing knowledge base")
        # Create a collection in Milvus
        collection = Collection(name=collection_name, schema=schema)

        # Create an index for the embedding field so we can search it later.
        # There are different index types, we will use IVF_FLAT for this example as it is simple and effective for small datasets.
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2", # L2 is the Euclidean distance, suitable for most use cases
            "params": {"nlist": 128} # nlist is the number of clusters, a good starting point is 100-200 for small datasets
        }
        collection.create_index(field_name="embedding", index_params=index_params)  
        collection.load()

    # Insert new data into the collection
    data = [
        exampleSourceDocuments,  # Texts
        embeddings.tolist()  # Embeddings
    ]
    insert_result = collection.insert(data)
    collection.flush()
    print(f"Inserted {len(insert_result.primary_keys)} documents into the collection.")
    print("Vector database initialized successfully.")



def queryVectorDb(query):
    print(f"Querying vector database for: {query}")
    # Connect to the Milvus server
    connections.connect("default", host="localhost", port="19530")

    # Access the collection we created earlier
    collection = Collection(name="sailing_knowledge_base")

    # Encode the query using the same model we used for the documents
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # Perform a similarity search
    results = collection.search(
        data=query_embedding.tolist(),  # The query embedding
        anns_field="embedding",  # The field we want to search
        param={"metric_type": "L2", "params": {"nprobe": 10}},  # Search parameters
        limit=5,  # Number of results to return
        output_fields=["text"],  # Fields to return in the results
    )
    print(f"Found {len(results[0])} results for the query.")
    return results

def query_with_rag(question):
    prompt = f"Question: {question}\nAnswer:"
    # Get response from Gemini
    answer = query_gemini(prompt)
    return answer


initVectorDb()  # Initialize the vector database and insert example documents

##
## Now we can perform a similarity search on the collection.
##

# Here is our example search query in plain text.
search_query = "How to win sailboat races"

# To search, we can now use the queryVectorDb function.
RAGresults = rag_query(search_query)
print ("RAG Results\n###########################################################")
# Print the results
print(f"Query: {search_query}")
print(f"Score: {RAGresults}\n")

print("************************************************************")
NoContexResults = query_with_rag(search_query)
print ("No RAG Results\n###########################################################")
print(f"Query: {search_query}")
print(f"Score: {NoContexResults}\n")


