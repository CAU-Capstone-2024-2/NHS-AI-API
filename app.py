from flask import Flask, request, jsonify
from openai import OpenAI
from contextual_vector_db import ContextualVectorDB
from dotenv import load_dotenv
import json
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

with open('data/doc.json', 'r') as f:
    transformed_dataset = json.load(f)

# ContextualVectorDB 초기화
db = ContextualVectorDB(name="test_db")
# 데이터 로드 (필요에 따라 load_data 메소드 사용)
db.load_data(transformed_dataset, parallel_threads=4)

# Flask 애플리케이션 초기화
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    session_id = data.get('sessionId', '')
    uid = data.get('uid', '')

    if not question or not session_id or not uid:
        return jsonify({"error": "질문, sessionId, uid를 모두 입력해주세요."}), 400

    # Immediately return 200 status code
    response = jsonify({"message": "응답이 성공적으로 처리되었습니다."})
    response.status_code = 200

    def process_question():
        try:
            # 질문과 관련된 상위 5개 문서 검색
            top_docs = db.search(query=question, k=5)

        # 관련 문서 내용 추출
        context = "\n\n".join([doc['metadata']['original_content'] for doc in top_docs])
        print(context)

        # GPT-4o에 전송할 메시지 생성
        messages = [
            {
                "role": "user",
                "content": f"""
You are an AI assistant designed to help elderly people by answering their questions in a clear and concise manner. Your task is to read a given document, understand the question asked, and provide a simple, easy-to-understand answer in Korean.

Here is the document you should use as a reference:

<document>
{context}
</document>

An elderly person has asked the following question:

<question>
{question}
</question>

Please follow these steps:

1. Carefully read and analyze the document.
2. Identify the key information in the document that relates to the question.
3. Formulate a concise answer that directly addresses the question.
4. Ensure your answer is simple and easy for an elderly person to understand.
5. If the document doesn't contain information to fully answer the question, state this clearly and provide whatever relevant information you can from the document.

When writing your response:
- Use simple, clear language
- Avoid complex terms or jargon
- Keep sentences short and to the point
- Be respectful and patient in your tone

Please provide your answer in Korean, formatted as plain text without any special formatting or tags. Your response should be concise, typically no more than 3-4 sentences, unless more detail is absolutely necessary to answer the question fully.

Begin your response now:
                """
            }
        ]

            # GPT-4o를 사용하여 답변 생성
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=8192,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "text"}
            )

            answer = gpt_response.choices[0].message.content.strip()

            # 외부 API에 응답 전송
            external_api_url = "http://100.99.151.44:1500/ask"
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "answer": answer
            }
            try:
                requests.post(external_api_url, json=external_api_data)
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

        except Exception as e:
            print(f"질문 처리 중 오류 발생: {str(e)}")

    # 비동기 작업 시작
    from threading import Thread
    Thread(target=process_question).start()

    return response

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=5056)
