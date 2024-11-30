import pickle
import os

def filter_vector_database(sentences_to_check):
    # Load original database
    with open('./data/contextual_vector_db.pkl', 'rb') as file:
        data = pickle.load(file)
    
    # Get all unique doc_ids where first chunk contains any of the sentences
    docs_to_remove = set()
    
    for i, meta in enumerate(data['metadata']):
        # Check only first chunks (chunk_id containing 'chunk_0')
        if 'chunk_0' in meta['chunk_id']:
            original_content = meta['original_content']
            # Check if any sentence is in the content
            if any(sentence.lower() in original_content.lower() for sentence in sentences_to_check):
                doc_id = meta['doc_id']
                docs_to_remove.add(doc_id)
    
    # Filter out the chunks from identified documents
    filtered_indices = [
        i for i, meta in enumerate(data['metadata'])
        if meta['doc_id'] not in docs_to_remove
    ]
    
    # Create new filtered data
    filtered_data = {
        'embeddings': [data['embeddings'][i] for i in filtered_indices],
        'metadata': [data['metadata'][i] for i in filtered_indices],
        'query_cache': data['query_cache']
    }
    
    # Save filtered database
    os.makedirs('./data', exist_ok=True)
    with open('./data/contextual_vector_db_filtered.pkl', 'wb') as file:
        pickle.dump(filtered_data, file)
    
    print(f"Removed all chunks from {len(docs_to_remove)} documents")
    print(f"Original database size: {len(data['metadata'])} chunks")
    print(f"Filtered database size: {len(filtered_data['metadata'])} chunks")

if __name__ == "__main__":
    sentences_to_check = ['sentence 1', 'sentence 2']
    filter_vector_database(sentences_to_check)
