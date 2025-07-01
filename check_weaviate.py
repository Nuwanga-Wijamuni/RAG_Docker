import weaviate
import weaviate.classes as wvc

def check_status():
    """Checks if the Weaviate instance is live and ready."""
    try:
        # The 'with' statement ensures the connection is always closed properly
        with weaviate.connect_to_local(host="localhost", port=8080) as client:
            print("✅ Successfully connected to Weaviate at http://localhost:8080")
            print(f"   Weaviate is ready: {client.is_ready()}")
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")

def list_schema():
    """Connects to Weaviate and prints the schema of all collections."""
    print("\n--- Database Schema ---")
    try:
        with weaviate.connect_to_local(host="localhost", port=8080) as client:
            collections = client.collections.list_all()
            if not collections:
                 print("No collections found in the database.")
            else:
                # --- FIX 1: Corrected Schema Logic ---
                # We now iterate through the names and get the full collection object
                for name in collections:
                    collection_obj = client.collections.get(name)
                    print(f"\nCollection: '{name}'")
                    # Now we can correctly access the properties
                    for prop in collection_obj.config.get().properties:
                        print(f"  - {prop.name}: {prop.data_type.value}") # Use .value for enum
    except Exception as e:
        print(f"❌ Could not retrieve schema: {e}")

def count_objects(collection_name: str):
    """Counts the total number of objects in a specific collection."""
    print(f"\n--- Object Count for '{collection_name}' ---")
    try:
        with weaviate.connect_to_local(host="localhost", port=8080) as client:
            if not client.collections.exists(collection_name):
                print(f"Collection '{collection_name}' does not exist.")
            else:
                collection = client.collections.get(collection_name)
                response = collection.aggregate.over_all(total_count=True)
                print(f"Total objects found: {response.total_count}")
    except Exception as e:
        print(f"❌ Could not count objects: {e}")

if __name__ == "__main__":
    print("Running Weaviate diagnostic script...")
    check_status()
    list_schema()
    count_objects("Document")