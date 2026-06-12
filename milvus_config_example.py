from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection

# --- Configuration via Code ---

# Define fields for the schema
# This demonstrates defining schema elements directly in Python code
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # Dimension of the vector
]

# Create a collection schema from the defined fields
schema = CollectionSchema(fields, "A simple schema for vector search.")

# Define collection name
collection_name = "my_vector_collection"

# --- MilvusLite Connection and Collection Creation ---

# Connect to MilvusLite (in-memory)
# This establishes a connection without needing a separate Milvus server
print("Connecting to MilvusLite...")
connections.connect("default", host="localhost", port="19530")
print("Connected.")

# Check if collection already exists, if so, drop it to start fresh
if utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' already exists. Dropping...")
    utility.drop_collection(collection_name)

# Create a new collection using the defined schema
# This is where the code-configured schema is applied
print(f"Creating collection '{collection_name}' with schema...")
collection = Collection(name=collection_name, schema=schema)
print("Collection created.")

# --- Example Data Insertion (Optional but good for demonstration) ---

# Prepare some dummy data
# In a real RAG scenario, these vectors would be embeddings of text/images
import random

num_entities = 10

# Generate random primary keys and vectors
ids = [i for i in range(num_entities)]
vectors = [[random.random() for _ in range(128)] for _ in range(num_entities)]

# Prepare data in the format Milvus expects
data = [
    ids,
    vectors
]

# Insert data into the collection
print(f"Inserting {num_entities} entities...")
collection.insert(data)
print("Data inserted.")

# Flush the collection to make data searchable
collection.flush()
print("Collection flushed.")

# --- Basic Search Example (Illustrates vector search capability) ---

# Create an index for the vector field to speed up searches
# This is also configured via code
index_params = {
    "metric_type": "L2",  # or "IP", "COSINE"
    "params": {"nlist": 1024}
}
print("Creating index...")
collection.create_index("vector", index_params)
print("Index created.")

# Load the collection into memory for searching
print("Loading collection into memory...")
collection.load()
print("Collection loaded.")

# Perform a search with a dummy query vector
query_vector = [random.random() for _ in range(128)]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

print("Performing a search...")
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param=search_params,
    limit=3,  # Number of nearest neighbors to return
    output_fields=["pk"] # Optionally return other fields
)

print("Search results:")
for hit in results[0]:
    print(f"  ID: {hit.id}, Distance: {hit.distance}")

# --- Cleanup ---

# Disconnect from Milvus
print("Disconnecting from Milvus...")
connections.disconnect("default")
print("Disconnected.")
