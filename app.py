import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from contextual_vector_db import ContextualVectorDB  # paste.txt에 정의된 클래스
import pickle

# 환경 변수에서 API 키 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# ContextualVectorDB 초기화
db = ContextualVectorDB(name="your_db_name")
# 데이터 로드 (필요에 따라 load_data 메소드 사용)
# 예: db.load_data(your_dataset, parallel_threads=4)

# Gemini 모델 생성
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    # safety_settings 등 추가 설정 가능
)

# Flask 애플리케이션 초기화
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    try:
        # 질문과 관련된 상위 5개 문서 검색
        top_docs = db.search(query=question, k=5)

        # 관련 문서 내용 추출
        context = "\n\n".join([doc['metadata']['contextualized_content'] for doc in top_docs])

        # Gemini에 전송할 프롬프트 생성
        prompt = f"다음 문서를 참고하여 질문에 답변해주세요:\n\n{context}\n\n질문: {question}\n답변:"

        # Gemini를 사용하여 답변 생성
        response = model.start_chat(history=[])
        response = response.send_message(prompt)
        answer = response.text.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=5000)
