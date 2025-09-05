# Milvus Sailing Knowledge Base

This project demonstrates how to set up and interact with a Milvus vector database to create a searchable knowledge example data. It uses `sentence-transformers` to generate embeddings for text documents and `pymilvus` to manage the Milvus collection, insert data, and perform similarity searches.

---

## Features

* **Milvus Integration:** Connects to a local Milvus instance in a docker container.
* **Sentence Embeddings:** Uses the `all-MiniLM-L6-v2` Sentence Transformer model to convert text into vector embeddings.
* **Vector Database Initialization:**
    * Defines a schema for a Milvus collection, including fields for ID, text content, and embeddings.
    * Creates a `sailing_knowledge_base` collection in Milvus.
    * Configures an `IVF_FLAT` index for efficient similarity searching.
    * Inserts a set of example sailing-related documents into the collection.
* **Similarity Search:** Performs vector similarity searches based on a natural language query, returning the most relevant documents from the knowledge base.

---

## Prerequisites

Before running this script, make sure you have the following:

* **Python 3.8+**
* **Docker Engine:** Ensure **Docker Engine** is installed and running on your system. This is crucial for running Milvus.

### Installing Python Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Start Milvus

Make sure your **Docker Engine** is running. Then, navigate to this directory containing your `docker-compose.yaml` and start the Milvus containers:

```bash
docker-compose up -d
```

This command starts Milvus and its dependencies (Etcd (a distributed key value store for metadata storage and access), MinIO (persistent storage for logs and index files)) in the background.

### 2. Run the Code

Execute the script from your terminal:

```bash
python vectordb-intro.py
```

The script will:
* **Initialize the vector database:** It will connect to Milvus, create the `sailing_knowledge_base` collection and its index, and insert the example sailing documents.
* **Perform a search:** It will then execute a predefined similarity search for the query "How to navigate in tidal waters?" and print the top matching documents along with their similarity scores.

### 3. View Vector DB Contents with Attu

You can use **Attu**, the Milvus management tool, to visualize and interact with your vector database. To start Attu and connect it to your running Milvus instance, use the following Docker command:

```bash
docker run -p 8000:3000 --network milvus -e HOST_URL=http://localhost:8000 -e MILVUS_URL=standalone:19530 zilliz/attu:latest
```

Once Attu is running, open your web browser and go to `http://localhost:8000`. When prompted for the Milvus address in the Attu UI, enter `standalone:19530`.

---

## Code Overview

### `initVectorDb()` Function

This function handles the setup of the Milvus collection and the initial data ingestion:

* **`exampleSourceDocuments`**: A list of strings representing the sailing knowledge articles to be indexed.
* **`SentenceTransformer('all-MiniLM-L6-v2')`**: Initializes the pre-trained model for generating embeddings.
* **`connections.connect("default", host="localhost", port="19530")`**: Establishes a connection to your Milvus instance.
* **`FieldSchema` and `CollectionSchema`**: Define the structure of your data within Milvus, specifying `id`, `text`, and `embedding` fields. The `embedding` field is defined as a `FLOAT_VECTOR` with `dim=384`, matching the output dimension of the `all-MiniLM-L6-v2` model.
* **`Collection(name="sailing_knowledge_base", schema=schema)`**: Creates the Milvus collection.
* **`collection.create_index(...)`**: Creates an `IVF_FLAT` index on the `embedding` field. This index allows for efficient approximate nearest neighbor (ANN) searches. `L2` (Euclidean distance) is used as the metric type.
* **`collection.load()`**: Loads the collection into memory, making it ready for operations like insertion and search.
* **`collection.insert(data)`**: Inserts the example documents and their corresponding embeddings into the collection.

### `queryVectorDb(query)` Function

This function performs a similarity search against the Milvus collection:

* **`connections.connect(...)`**: Reconnects to the Milvus instance (if not already connected).
* **`Collection(name="sailing_knowledge_base")`**: Accesses the previously created collection.
* **`model.encode([query])`**: Generates an embedding for the input `query` using the same Sentence Transformer model.
* **`collection.search(...)`**: Executes the similarity search.
    * `data`: The embedding of the query.
    * `anns_field`: Specifies that the search should be performed on the `embedding` field.
    * `param`: Contains search-specific parameters, including `metric_type` (L2) and `nprobe` (number of clusters to search).
    * `limit`: The maximum number of nearest neighbors to return.
    * `output_fields`: Specifies to return the `text` field of the matching documents.

### Main Execution Block

The script directly calls `initVectorDb()` (commented out by default to prevent re-initialization on every run) and then `queryVectorDb()` with a sample query, printing the results.

---

## Extending the Knowledge Base

To add more documents to your `sailing_knowledge_base`:

1.  Add new strings to the `exampleSourceDocuments` list in `initVectorDb()`.
2.  If the collection already exists, you might need to drop it first (`collection.drop()`) or add logic to check for its existence before creation to avoid errors. Then, uncomment and re-run `initVectorDb()` from `milvus_kb.py`.

---