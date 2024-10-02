import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from ContextualVectorDB import ContextualVectorDB

app = Flask(__name__)

# Load your ContextualVectorDB instance (replace with your actual loading logic)
vector_db = ContextualVectorDB.load_db("./data/your_db_name/contextual_vector_db.pkl") 

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
generation_config = {
    "temperature": 0.7,  # Adjust as needed
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,  # Adjust as needed
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Missing question'}), 400

    # Search for relevant documents
    results = vector_db.search(question, k=5) 

    # Prepare context for Gemini
    context = ""
    for result in results:
        context += f"## {result['metadata']['contextualized_content']}\n\n{result['metadata']['original_content']}\n\n"

    # Generate answer using Gemini
    chat_session = model.start_chat(history=[])
    prompt = f"### Question:\n{question}\n\n### Context:\n{context}\n\n### Answer:"
    response = chat_session.send_message(prompt)

    return jsonify({'answer': response.text})

if __name__ == '__main__':
    app.run(debug=True)
