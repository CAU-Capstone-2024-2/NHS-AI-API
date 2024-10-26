from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI
from contextual_vector_db import ContextualVectorDB
from dotenv import load_dotenv
import json
import aiohttp
from typing import Optional

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

# FastAPI 애플리케이션 초기화
app = FastAPI()

class QuestionRequest(BaseModel):
    sessionId: str
    uid: str
    question: str

@app.post('/qsmaker')
async def make_questions(request: QuestionRequest, background_tasks: BackgroundTasks):
    if not request.question or not request.sessionId or not request.uid:
        raise HTTPException(status_code=400, detail="sessionId, uid, question를 모두 입력해주세요.")

    async def process_clarifying_questions(session_id: str, uid: str, question: str):
        try:
            # GPT-4o를 사용하여 명확한 질문 생성
            messages = [
                {
                    "role": "user",
                    "content": f"""Given the following question from an elderly person, generate three more specific questions in Korean that include a question from an elderly person.:
Question: {question}

Example)
Question: 만성 C형간염을 어떨 때 치료해야하나요?
Answer: "clarifying_questions":["1. 만성 C형간염 치료가 필요한 증상이나 징후는 무엇인가요?","2. 만성 C형간염 치료를 시작하기에 적절한 시기는 언제인가요?","3. 만성 C형간염을 치료하지 않으면 어떤 위험이 있을까요?"]

Please generate questions that are:
1. Simple and easy to understand
2. Directly related to the original question
3. Help clarify any ambiguous parts of the question
"""
                }
            ]

            gpt_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clarifying_questions",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "clarifying_questions": {
                                    "type": "array",
                                    "description": "A list of three questions generated to clarify the question.",
                                    "items": {
                                        "type": "string",
                                        "description": "Each clarifying question."
                                    }
                                }
                            },
                            "required": ["clarifying_questions"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            # Extract the clarifying questions from the response
            response_json = json.loads(gpt_response.choices[0].message.content)
            clarifying_questions = response_json["clarifying_questions"]
            print(clarifying_questions)
            # 외부 API에 응답 전송
            external_api_url = "http://100.99.151.44:1500/api/answer"
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "clarifying_questions": clarifying_questions,
                "status_code": 200
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(external_api_url, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

        except Exception as e:
            error_message = f"질문 처리 중 오류 발생: {str(e)}"
            print(error_message)
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "answer": error_message,
                "status_code": 500
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(external_api_url, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

    # Add the background task and return response
    background_tasks.add_task(process_clarifying_questions, request.sessionId, request.uid, request.question)
    return {"message": "응답이 성공적으로 처리되었습니다."}

@app.post('/ask')
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    if not request.question or not request.sessionId or not request.uid:
        raise HTTPException(status_code=400, detail="sessionId, uid, question를 모두 입력해주세요.")

    async def process_question(session_id: str, uid: str, question: str):
        try:
            # 질문과 관련된 상위 5개 문서 검색
            top_docs = db.search(query=question, k=5)

            # 관련 문서 내용 추출
            context = ""
            for doc in top_docs:
                context += f"문맥:{doc['metadata']['contextualized_content']}{doc['metadata']['original_content']}\n\n"
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
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=8192,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "poster_template",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "template_type": {
                                    "type": "string", 
                                    "description": "The type of the poster template used.",
                                    "enum": ["qna__square_single"]
                                },
                                "content": {
                                    "anyOf": [{
                                        "type": "object",
                                        "properties": {
                                            "question": {
                                                "type": "string",
                                                "description": "The first question for template 1."
                                            },
                                            "answer": {
                                                "type": "string",
                                                "description": "The answer to the first question."
                                            }
                                        },
                                        "required": ["question", "answer"],
                                        "additionalProperties": False
                                    }]
                                }
                            },
                            "required": ["template_type", "content"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            # Get the raw JSON response
            raw_response = gpt_response.choices[0].message.content
            print(raw_response)
            # 외부 API에 응답 전송
            external_api_url = "http://100.99.151.44:1500/api/answer"
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "answer": raw_response,
                "status_code": 200  # Successful processing
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(external_api_url, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

        except Exception as e:
            error_message = f"질문 처리 중 오류 발생: {str(e)}"
            print(error_message)
            # Send error status to external API
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "answer": error_message,
                "status_code": 500  # Internal server error
            }
            try:
                response = requests.post(external_api_url, json=external_api_data)
                print(response.text)
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

    # 비동기 작업 시작
    # Add the background task and return response
    background_tasks.add_task(process_question, request.sessionId, request.uid, request.question)
    return {"message": "응답이 성공적으로 처리되었습니다."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5056)
