from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType


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
        "Winning sailboat races involves a combination of boat speed, strategic decision-making, and understanding the rules. Key tactics include starting line strategy, tacking and gybing efficiently, understanding wind shifts, and exploiting current. Knowing the Racing Rules of Sailing (RRS) is paramount.",
        "Proper anchoring techniques are essential for securing your boat. Different anchor types (Danforth, CQR, Bruce, Rocna) perform best on various bottom compositions (sand, mud, rock). Factors like scope, depth, and swing room must be considered. Always test your anchor's set.",
        "Sailing can be a fantastic family activity. Involve children in age-appropriate tasks, ensure their safety with life jackets and safety netting, and make the experience fun with games and destinations. Pack plenty of snacks and sun protection. Start with shorter trips and gradually increase duration."
    ]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(exampleSourceDocuments)

    connections.connect("default", host="localhost", port="19530")

    # Lets define the fields in our database collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]

    # Next create a schema by using the fields defined above
    schema = CollectionSchema(fields, description="Sailing knowledge base")
    # Now we can create a collection in Milvus
    collection = Collection(name="sailing_knowledge_base", schema=schema)

    # Create an index for the embedding field so we can search it later.
    # There are different index types, we will use IVF_FLAT for this example as it is simple and effective for small datasets.
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2", # L2 is the Euclidean distance, suitable for most use cases
        "params": {"nlist": 128} # nlist is the number of clusters, a good starting point is 100-200 for small datasets
    }
    collection.create_index(field_name="embedding", index_params=index_params)  

    collection.load() #this loads the collection into memory even though there is no data yet

    # Now we are ready to insert data into the collection.
    # First we need to prepare the data in the format what was defined in the schema.
    data = [
        exampleSourceDocuments,  # Texts
        embeddings.tolist()  # Embeddings
    ]
    # Insert the data into the collection
    insert_result = collection.insert(data)
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


#initVectorDb()  # Initialize the vector database and insert example documents

##
## Now we can perform a similarity search on the collection.
##

# Here is our example search query in plain text.
search_query = "How to navigate in tidal waters?"

# To search, we can now use the queryVectorDb function.
results = queryVectorDb(search_query)
print ("Results\n###########################################################")
# Print the results
for result in results[0]:
    print(f"Query: {search_query}")
    print(f"Result document: {result.entity.text}")
    print(f"Score: {result.distance:.4f}\n")
