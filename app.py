import os
from flask import Flask, request, jsonify
from openai import OpenAI
from contextual_vector_db import ContextualVectorDB
from dotenv import load_dotenv
import json

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

with open('data/doc.json', 'r') as f:
    transformed_dataset = json.load(f)

# ContextualVectorDB 초기화
db = ContextualVectorDB(name="test_db")
# 데이터 로드 (필요에 따라 load_data 메소드 사용)
db.load_data(transformed_dataset, parallel_threads=1)

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
        context = "\n\n".join([doc['metadata']['original_content'] for doc in top_docs])
        print(context)

        # GPT-4o에 전송할 메시지 생성
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Respond to a question from an elderly person using a provided document as a reference. The response should be concise and written in Korean, formatted in a way that is easy for elderly individuals to understand."
                    }
                ]
            },
            {
                "role": "user",
                "content": f"문서: {context}\n\n질문: {question}"
            }
        ]

        # GPT-4o를 사용하여 답변 생성
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=8192,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "text"}
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=5056)
