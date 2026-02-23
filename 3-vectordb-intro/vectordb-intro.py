"""
ChromaDB Sailing Knowledge Base - Vector Database Intro

This script demonstrates the core concepts of a vector database using ChromaDB.
It builds a small sailing knowledge base by storing text documents as vector
embeddings, then performs semantic similarity searches against them.

Key concepts covered:
  - Creating and persisting a ChromaDB collection
  - Using ChromaDB's built-in embedding model (all-MiniLM-L6-v2 via onnxruntime)
    to automatically convert text into vector embeddings on insert and query
  - Performing a similarity search using natural language and interpreting
    the distance scores returned (lower = more similar)
"""

import chromadb

DATABASE_FILE_PATH = "./chroma_db_data"  # Path where ChromaDB will store its data


def initVectorDb():
    print("Initializing vector database...")
    exampleSourceDocuments = [
        "Dinghy sailing is an accessible and exhilarating way to get started on the water. Small, nimble boats like Optimists, Lasers, and 420s are popular choices for learning the basics of rigging, steering, and tacking. Safety is paramount, so always wear a life jacket and understand basic right-of-way rules. Many sailing clubs offer introductory courses.",
        "Navigating along a coastline requires attention to charts, tides, currents, and weather. Understanding buoyage systems, using GPS effectively, and plotting courses are crucial skills. Always carry paper charts as a backup to electronic navigation. Be aware of safe depths and potential hazards like rocks and shoals.",
        "Achieving optimal sail trim is key to maximizing boat speed and efficiency. This involves adjusting the mainsheet, jib sheets, vang, outhaul, and cunningham. Understanding wind angles and sail shape variations for different points of sail (upwind, reaching, downwind) is fundamental. Practice and feel are essential for mastery.",
        "Offshore cruising offers the ultimate freedom and adventure. Planning for long passages involves provisioning, checking all systems, and monitoring long-range weather forecasts. Self-sufficiency is vital, as help may be days away. Enjoying remote anchorages and starry nights at sea are unforgettable experiences.",
        "Accurate marine weather forecasting is critical for safe sailing. Learn to interpret synoptic charts, GRIB files, and local weather reports. Pay attention to wind speed and direction and wave height, squalls, and fog. Marine VHF radio and satellite communicators are valuable tools for receiving updates.",
        "Mastering a few essential sailing knots is fundamental for any sailor. Key knots include the bowline for forming a secure loop, the cleat hitch for tying to a cleat, the sheet bend for joining two ropes, and the figure-eight knot as a stopper knot. Practice makes perfect for tying them quickly and correctly.",
        "Regular maintenance of your sailboat's auxiliary engine is crucial for reliability, especially on longer trips. This includes checking oil levels, fuel filters, impellers, and belts. Learning basic troubleshooting can save a voyage. Always carry spare parts for common issues.",
        "Winning sailboat races involves a combination of boat speed, strategic decision-making, and understanding the rules. Key tactics include starting line strategy, sail trim, boat handling by tacking and gybing efficiently, understanding wind shifts, and exploiting current. Knowing the Racing Rules of Sailing (RRS) is paramount.",
        "Proper anchoring techniques are essential for securing your boat. Different anchor types (Danforth, CQR, Bruce, Rocna) perform best on various bottom compositions (sand, mud, rock). Factors like scope, depth, and swing room must be considered. Always test your anchor's set.",
        "Sailing can be a fantastic family activity. Involve children in age-appropriate tasks, ensure their safety with life jackets and safety netting, and make the experience fun with games and destinations. Pack plenty of snacks and sun protection. Start with shorter trips and gradually increase duration."
    ]

    # PersistentClient stores the database on disk so data survives between runs
    client = chromadb.PersistentClient(path=DATABASE_FILE_PATH)

    # get_or_create_collection avoids errors if the collection already exists
    # ChromaDB uses its built-in embedding model (all-MiniLM-L6-v2 via onnxruntime) by default
    collection = client.get_or_create_collection(name="sailing_knowledge_base")

    # Assume the collection is already initialized if it contains the same number of documents as our example set
    if(collection.count() == exampleSourceDocuments.__len__()):
        print("Collection already initialized with example documents.")
        return  
    
    # Add documents - ChromaDB automatically generates embeddings for them
    collection.add(
        documents=exampleSourceDocuments,
        ids=[str(i) for i in range(len(exampleSourceDocuments))]
    )

    print(f"Inserted {len(exampleSourceDocuments)} documents into the collection.")
    print("Vector database initialized successfully.")


def queryVectorDb(query):
    print(f"Querying vector database for: {query}")

    client = chromadb.PersistentClient(path=DATABASE_FILE_PATH)
    collection = client.get_collection(name="sailing_knowledge_base")

    # ChromaDB automatically embeds the query text using the same built-in model
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    print(f"Found {len(results['documents'][0])} results for the query.")
    return results


initVectorDb()  # Initialize the vector database and insert example documents

##
## Now we can perform a similarity search on the collection.
##

# Here is our example search query in plain text.
search_query = "How to win sailboat races, what do i need to know and master?"

# To search, we can now use the queryVectorDb function.
results = queryVectorDb(search_query)
print("Results and scores (smaller distance is better)\n###########################################################")

# Print the results
for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"Result document: {doc}")
    print(f"Score: {distance:.4f}\n")
