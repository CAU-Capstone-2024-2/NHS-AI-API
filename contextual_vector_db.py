import os
import pickle
import json
import numpy as np
import voyageai
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

class ContextualVectorDB:
    def __init__(self, name: str):
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"
        self.checkpoint_path = f"./data/{name}/checkpoint.pkl"
        self.custom_info_db_path = f"./data/{name}/custom_info_db.pkl"
        self.acute_db_path = f"./data/{name}/acute_ql_db.pkl"
        
        self.token_lock = threading.Lock()
        self.custom_info_embeddings = []
        self.custom_info_metadata = []
        self.load_custom_information()

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        prompt = f"""
        <document>
        {doc}
        </document>

        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk}
        </chunk>

        Please give a short succinct context in Korean to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        completion = self.client.chat.completions.create(
            model="google/gemini-flash-1.5-8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content, None

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        # Check for checkpoint
        checkpoint = self._load_checkpoint()
        if checkpoint:
            texts_to_embed = checkpoint['texts_to_embed']
            metadata = checkpoint['metadata']
            processed_chunks = checkpoint['processed_chunks']
            print(f"Resuming from checkpoint with {len(processed_chunks)} processed chunks")
        else:
            texts_to_embed = []
            metadata = []
            processed_chunks = set()

        total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc, chunk):
            if 'doc_0' in chunk['chunk_id']:
                # Skip situate_context for doc_0
                contextualized_text = ""
            else:
                # For each chunk, produce the context
                contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])
            
            with self.token_lock:
                return {
                # Append the context to the original text chunk
                'text_to_embed': chunk['content'] if chunk['chunk_id'] == 'doc_0' else f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'original_uuid': doc['original_uuid'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc['chunks']:
                    chunk_id = f"{doc['doc_id']}_{chunk['chunk_id']}"
                    if chunk_id not in processed_chunks:
                        futures.append(executor.submit(process_chunk, doc, chunk))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])
                processed_chunks.add(f"{result['metadata']['doc_id']}_{result['metadata']['chunk_id']}")
                
                # Save checkpoint every 10 chunks
                if len(processed_chunks) % 10 == 0:
                    self._save_checkpoint(texts_to_embed, metadata, processed_chunks)

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        self.embeddings = []
        self.metadata = data

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            attempt = 0
            while True:
                try:
                    result = self.voyage_client.embed(
                        batch_texts,
                        model="voyage-3"
                    ).embeddings
                    self.embeddings.extend(result)
                    break  # Exit the retry loop if successful
                except Exception as e:
                    attempt += 1
                    print(f"Embedding failed for batch {i // batch_size + 1}. Retrying in 60 seconds... (Attempt {attempt})")
                    time.sleep(60)  # Wait for 60 seconds before retryin

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-3").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            print(similarities[idx])
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

    def load_acute_data(self):
        if os.path.exists(self.acute_db_path):
            with open(self.acute_db_path, "rb") as file:
                data = pickle.load(file)
                self.acute_embeddings = data["embeddings"]
                self.acute_metadata = data["metadata"]
            return

        try:
            acute_questions = []
            acute_metadata = []
            with open('./data/acute_ql_dataset.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    acute_questions.append(data["question"])
                    acute_metadata.append(data)

            if acute_questions:
                # Process in batches of 128
                batch_size = 128
                embeddings = []
                
                for i in range(0, len(acute_questions), batch_size):
                    batch = acute_questions[i:i + batch_size]
                    batch_embeddings = self.voyage_client.embed(batch, model="voyage-3").embeddings
                    embeddings.extend(batch_embeddings)
                
                data = {
                    "embeddings": embeddings,
                    "metadata": acute_metadata
                }
                
                os.makedirs(os.path.dirname(self.acute_db_path), exist_ok=True)
                with open(self.acute_db_path, "wb") as file:
                    pickle.dump(data, file)
                
                self.acute_embeddings = embeddings
                self.acute_metadata = acute_metadata
        except FileNotFoundError:
            print("Acute dataset file not found")
            return

    def search_acute(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not hasattr(self, 'acute_embeddings') or not self.acute_embeddings:
            return []

        query_embedding = self.voyage_client.embed([query], model="voyage-3").embeddings[0]
        similarities = np.dot(self.acute_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.acute_metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def search_acute_exact(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for exact match in acute questions"""
        if not hasattr(self, 'acute_metadata'):
            return None
            
        # Normalize query for comparison
        query = query.strip().lower()
        
        # Look for exact match
        for meta in self.acute_metadata:
            if meta["question"].strip().lower() == query:
                return {
                    "metadata": meta,
                    "similarity": 1.0
                }
        
        # If no exact match, fall back to similarity search
        results = self.search_acute(query, k=1)
        return results[0] if results else None

    def load_custom_information(self):
        if os.path.exists(self.custom_info_db_path):
            with open(self.custom_info_db_path, "rb") as file:
                data = pickle.load(file)
                self.custom_info_embeddings = data["embeddings"]
                self.custom_info_metadata = data["metadata"]
            return

        try:
            with open('./data/custom_information.json', 'r', encoding='utf-8') as file:
                custom_info = json.load(file)
        except FileNotFoundError:
            print("Custom information file not found")
            return

        texts_to_embed = []
        metadata = []
        custom_info = custom_info["custom_information"]
        for disease_data in custom_info:
            for disease_name, sentences in disease_data.items():
                for item in sentences:
                    texts_to_embed.append(item["sentence"])
                    metadata.append({
                        "disease": disease_name,
                        "index": item["index"],
                        "sentence": item["sentence"],
                        "original": item["original"],
                        "img": item["img"],
                        "img_big": item["img_big"]
                    })

        if texts_to_embed:
            embeddings = self.voyage_client.embed(texts_to_embed, model="voyage-3").embeddings
            
            data = {
                "embeddings": embeddings,
                "metadata": metadata
            }
            
            os.makedirs(os.path.dirname(self.custom_info_db_path), exist_ok=True)
            with open(self.custom_info_db_path, "wb") as file:
                pickle.dump(data, file)
            
            self.custom_info_embeddings = embeddings
            self.custom_info_metadata = metadata

    def get_custom_information(self, info: str, penalize_indices: list[str] = None) -> dict:
        import re
        import numpy as np
        
        if penalize_indices is None:
            penalize_indices = []
            
        # Extract disease names
        disease_match = re.search(r'<disease>(.*?)</disease>', info)
        if not disease_match:
            return {}
            
        diseases = [d.strip() for d in disease_match.group(1).split(',')]
        
        # Map Korean disease names to English
        disease_map = {
            "고혈압": "hypertension",
            "이상지질혈증": "dyslipidemia",
            "당뇨병": "diabetes"
        }
        
        # Convert to English names
        target_diseases = [disease_map.get(d) for d in diseases if disease_map.get(d)]
        
        if not target_diseases:
            return {}
            
        # Get relevant metadata indices
        relevant_indices = [
            i for i, meta in enumerate(self.custom_info_metadata)
            if meta["disease"] in target_diseases
        ]
        
        if not relevant_indices:
            return {}
            
        # Calculate initial probabilities (equal weights)
        probabilities = np.ones(len(relevant_indices)) / len(relevant_indices)
        
        # Get info embedding
        info_embedding = self.voyage_client.embed([info], model="voyage-3").embeddings[0]
        
        # Calculate similarities and update probabilities
        similarities = []
        for idx in relevant_indices:
            similarity = np.dot(self.custom_info_embeddings[idx], info_embedding)
            # Apply penalty if index is in penalize_indices
            if str(self.custom_info_metadata[idx]["index"]) in penalize_indices:
                similarity *= 0.1  # Strong penalty factor
            similarities.append(similarity)
        
        # Normalize similarities to probabilities
        similarities = np.array(similarities)
        similarities = np.exp(similarities) / np.sum(np.exp(similarities))
        
        # Sample based on probabilities
        selected_idx = np.random.choice(relevant_indices, p=similarities)
        
        # Return both img_big URL and index
        return {
            "img_url": self.custom_info_metadata[selected_idx]["img_big"],
            "index": str(self.custom_info_metadata[selected_idx]["index"])
        }

    def _save_checkpoint(self, texts_to_embed, metadata, processed_chunks):
        checkpoint_data = {
            'texts_to_embed': texts_to_embed,
            'metadata': metadata,
            'processed_chunks': processed_chunks
        }
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return None
        return None
