import weaviate
from langchain.text_splitter import CharacterTextSplitter
import ollama
import time
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="weaviate.warnings", message="Dep016: Python client v3 `weaviate.Client(...)` connections and methods are deprecated.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>")

# Initialize Weaviate client
print("Initializing Weaviate client...")
weaviate_client = weaviate.Client("http://localhost:8080")

# Custom embedding function using Ollama
def get_ollama_embedding(text):
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return embedding['embedding']

# Load and print the document
print("Fetching file")
with open("aspirenex.txt", "r") as file:
    full_text = file.read()
    print("\nFull text content:")
    print(full_text)
    print("\n" + "="*50 + "\n")

# Chunk the document
print("Chunking document...")
start_time1 = time.time()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=75, separator='\n', strip_whitespace=False)
chunks = text_splitter.split_text(full_text)
end_time1 = time.time()
print(f"\nText chunked in {end_time1 - start_time1:.2f} seconds.")
print(f"Document split into {len(chunks)} chunks.")
print("\n")

print("\nPrinting chunks:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-"*50)

# Define Weaviate schema
class_name = "AspireTest2"
class_obj = {
    "class": class_name,
    "vectorizer": "none",  # We'll provide our own vectors
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]}
    ]
}

# Create the schema in Weaviate
print(f"Creating schema in Weaviate for class: {class_name}")
weaviate_client.schema.create_class(class_obj)

# Upload vectors to Weaviate
print(f"Uploading vectors to Weaviate...")
start_time = time.time()
batch_size = 100
with weaviate_client.batch as batch:
    batch.batch_size = batch_size
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}", end="\r")
        embedding = get_ollama_embedding(chunk)
        data_object = {
            "content": chunk,
            "source": "aspirenex.txt"
        }
        batch.add_data_object(
            data_object=data_object,
            class_name=class_name,
            vector=embedding,
            uuid=uuid.uuid4()
        )
end_time = time.time()
print(f"\nVectors uploaded to Weaviate in {end_time - start_time:.2f} seconds.")
print("\n")