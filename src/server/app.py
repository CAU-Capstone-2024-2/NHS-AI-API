import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from openai import OpenAI
from .vector_db import ContextualVectorDB
from dotenv import load_dotenv
import json
import aiohttp
from typing import Optional
import os
import asyncio
# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()
acute_client = OpenAI(
    base_url=os.getenv("ACUTE_ADRES"),
    api_key=os.getenv("ACUTE_API_KEY")
)

with open('data/doc.json', 'r') as f:
    transformed_dataset = json.load(f)

# ContextualVectorDB 초기화
db = ContextualVectorDB(name="test_db")
# 데이터 로드 (필요에 따라 load_data 메소드 사용)
db.load_data(transformed_dataset, parallel_threads=4)
db.load_acute_data()

# FastAPI 애플리케이션 초기화
app = FastAPI()

class QuestionRequest(BaseModel):
    sessionId: str
    uid: str
    question: str
    info: str = ""
    isAcute: bool = False

class CustomInformationRequest(BaseModel):
    info: str = Field(..., description="Information string containing disease tags")
    index: list[str] = Field(default=[], description="List of indices to penalize")

# Define the external API URL as a constant
EXTERNAL_API_URL = "http://100.119.71.36:1500/api/answer"
DESIRED_TERMS = ['급성', '급성고환염', '급성 합병증', '당뇨병케토산증', '저혈당', '당뇨병 합병증(급성 합병증)', '당뇨병 합병증(급성 합병증_저혈당)', '저혈당' '급성부고환염', '급성 간부전', '급성 바이러스 위장관염', '급성신손상(소아)', '급성 세균성 장염', '노로바이러스', '심금연' '심낭염(급성 심낭염)', '당뇨병 합병증(급성 합병증_당뇨병케토뇨증', '고혈당고삼투질상태)', '급성 심근경색증', '급성 충수염', '급성호흡기바이러스감염증', '급성호흡곤란증후군', '심부전', '부정맥', '심장 판막 질환', '대동맥 박리', '심실중격결손증', '동맥관 개존증', '심방중격결손증', '폐색전증', '감염성 심내막염', '심낭염', '고혈압성 심장질환', '협심증', '폐렴', '만성폐쇄성폐질환', '기흉', '부신부전증', '갑상선 기능 항진증', '갑상선 기능 저하증', '갈색세포종', '뇌졸중', '뇌전증', '뇌수막염', '뇌하수체 기능 저하증', '패혈증', '중증열성혈소판감소증후군', '말라리아', '레지오넬라증', '일본뇌염', '광견병', '파상풍', '디프테리아', '백일해', '비브리오 패혈증', '아나필락시스', '독극물 섭취', '영아돌연사증후군', '췌장염', '장결핵', '샤가스병', '바이러스성 출혈열', '심장마비']

@app.post('/qsmaker')
async def make_questions(request: QuestionRequest, background_tasks: BackgroundTasks):
    if not request.question or not request.sessionId or not request.uid:
        raise HTTPException(status_code=400, detail="sessionId, uid, question를 모두 입력해주세요.")

    async def check_acute(question: str) -> bool:
        try:
            prompt_question = f"다음 건강 정보 관련 질문이 [급성고환염, 당뇨병 합병증(급성 합병증), 당뇨병 합병증(급성 합병증_저혈당), 급성부고환염, 급성 간부전, 급성 바이러스 위장관염, 급성신손상(소아), 급성 세균성 장염, 심낭염(급성 심낭염), 당뇨병 합병증(급성 합병증_당뇨병케토산증, 고혈당고삼투질상태), 급성 심근경색증, 급성 충수염, 급성호흡기바이러스감염증, 급성호흡곤란증후군, 심부전, 부정맥, 심장 판막 질환, 대동맥 박리, 심실중격결손증, 동맥관 개존증, 심방중격결손증, 폐색전증, 감염성 심내막염, 심낭염, 고혈압성 심장질환, 협심증, 폐렴, 만성폐쇄성폐질환, 기흉, 부신부전증, 갑상선 기능 항진증, 갑상선 기능 저하증, 갈색세포종, 뇌졸중, 뇌전증, 뇌수막염, 뇌하수체 기능 저하증, 패혈증, 중증열성혈소판감소증후군, 말라리아, 레지오넬라증, 일본뇌염, 응급, 광견병, 파상풍, 디프테리아, 백일해, 비브리오 패혈증, 아나필락시스, 독극물 섭취, 영아돌연사증후군, 췌장염, 장결핵, 샤가스병, 바이러스성 출혈열] 카테고리 안에 속한다며 True 속하지 않는다면 False을 출력하세요. 다른 내용 없이 True 또는 False만을 출력하세요.: {question}"

            for term in DESIRED_TERMS:
                if term in question:
                    return True


            acute_completion = acute_client.with_options(timeout=2).chat.completions.create(
                model="mldljyh/nhs_1.5b_1_r16_merged_t2",
                messages=[
                    {
                        "role": "system",
                        "content": ""
                    },
                    {
                        "role": "user",
                        "content": prompt_question
                    }
                ],
                temperature=0,
                top_p=0.1
            )
            response = acute_completion.choices[0].message.content.strip()
            print(response)
            return "true" in response.lower()
        
        except Exception as e:
            print(f"Acute API error: {str(e)}, falling back to gpt-4o-mini")
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": ""
                        },
                        {
                            "role": "user",
                            "content": prompt_question
                        }
                    ],
                    temperature=0,
                    top_p=0.1
                )
                response = completion.choices[0].message.content.strip()
                return "true" in response.lower()
            except Exception as e:
                print(f"GPT-4o-mini error: {str(e)}")
                return False

    async def process_clarifying_questions(session_id: str, uid: str, question: str):
        # Check if it's an acute question
        is_acute = await check_acute(question)
        print(is_acute)
        if is_acute:
            acute_results = db.search_acute(question)
            if acute_results:
                clarifying_questions = [result["metadata"]["question"] for result in acute_results]
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "clarifying_questions": clarifying_questions,
                    "status_code": 212
                }
                print(clarifying_questions)
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
                return
        try:
            # 질문과 관련된 문서 검색
            top_docs = db.search(query=question, k=3)

            # 유사도가 0.2 미만인 경우 답변 불가능으로 처리
            if not top_docs or top_docs[0]['similarity'] < 0.30:
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "answer": "안녕하세요! 저는 현재 건강 정보를 중심으로 돕고 있어요. 건강과 관련된 질문을 구체적으로 해주시면 더 자세한 도움을 드릴 수 있어요!",
                    "status_code": 423
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
                return

            # 관련 문서 내용 추출
            context = ""
            for doc in top_docs:
                if doc['metadata']['contextualized_content'] == "":
                    context += f"{doc['metadata']['original_content']}\n\n"
                else:
                    context += f"문맥:{doc['metadata']['contextualized_content']}{doc['metadata']['original_content']}\n\n"
                
            # GPT-4o를 사용하여 명확한 질문 생성
            messages = [
                {
                    "role": "user",
                    "content": f"""You are an AI assistant for a chatbot service that provides health information to the elderly. Your task is to create clarifying topics based on a user's initial query and reference documents.

Here is the user's question:
<user_question>
{question}
</user_question>

Here are the reference documents provided for answering health-related questions:
<reference_documents>
{context}
</reference_documents>

Your task is to create three specific, concise topics in Korean that can help find relevant health information by clarifying the user's question. These questions should be aimed at finding appropriate documents to answer the user's query, not at gathering more information from the user.

Guidelines for creating topics:
1. Topics should be in Korean.
2. Topics should be specific and relevant to the user's initial query.
3. Topics should focus on finding information within the reference documents.
4. Please modify the question to generate it with the following specific topic with emojis.: "🩺 당뇨병 증상과 진단 기준"
5. Do not make topics directed at the elderly; instead, frame them as if you're searching for information.
6. Avoid yes/no topics; use open-ended topics that can lead to more detailed information.
7. Do not create topics that directly ask if there is a document
8. Do not create topics related to the following.: [{', '.join(DESIRED_TERMS)}]

If the user's question is not related to health information, do not create any questions. In this case, provide an empty list.

Present your output in the following format:
[List your three topics in Korean here, one per line. If the query is not health-related, leave this section empty.]

Remember, do not number the topics, and ensure they are written in Korean.
Use emojis that fit well with the topic, in addition to the ones used in the example.
Please concisely write only the topic, not in question form.
"""
                }
            ]
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
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
            print(gpt_response.choices[0].message.content)
            response_json = json.loads(gpt_response.choices[0].message.content)
            print(response_json)
            clarifying_questions = response_json["clarifying_questions"]
            print(clarifying_questions)

            # 질문 리스트가 비어있는 경우 답변 불가능으로 처리
            if not clarifying_questions:
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "answer": "안녕하세요! 저는 현재 건강 정보를 중심으로 돕고 있어요. 건강과 관련된 질문을 구체적으로 해주시면 더 자세한 도움을 드릴 수 있어요!",
                    "status_code": 423
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        try:
                            result = await response.json()
                            print("Response:", result)
                        except json.JSONDecodeError as e:
                            print("JSON Parsing Error:", e)
                            print("Raw response:", await response.text())
                return

            # If clarifying_questions is not empty, check acute in parallel
            if clarifying_questions:
                acute_results = await asyncio.gather(
                    *[check_acute(q) for q in clarifying_questions[:3]]
                )
                print(acute_results)
                # If two or more questions are acute, proceed as if is_acute is True
                if sum(acute_results) == 3:
                    acute_search_results = db.search_acute(clarifying_questions[0])
                    if acute_search_results:
                        clarifying_questions = [result["metadata"]["question"] for result in acute_search_results]
                        external_api_data = {
                            "sessionId": session_id,
                            "uid": uid,
                            "clarifying_questions": clarifying_questions,
                            "status_code": 212
                        }
                        print(external_api_data)
                        async with aiohttp.ClientSession() as session:
                            async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                                print(await response.text())
                        return

            # 외부 API에 응답 전송
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "clarifying_questions": clarifying_questions,
                "status_code": 211
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
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
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

    # Add the background task and return response
    background_tasks.add_task(process_clarifying_questions, request.sessionId, request.uid, request.question)
    return {"message": "응답이 성공적으로 처리되었습니다."}

@app.post('/custom_information')
async def get_custom_information(request: CustomInformationRequest):
    result = db.get_custom_information(request.info, request.index)
    if not result:
        raise HTTPException(status_code=404, detail="No relevant information found")
    return result

@app.post('/ask')
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    if not request.question or not request.sessionId or not request.uid:
        raise HTTPException(status_code=400, detail="sessionId, uid, question를 모두 입력해주세요.")

    async def process_acute_question(session_id: str, uid: str, question: str):
        try:
            acute_results = db.search_acute_exact(question)
            if acute_results:
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "answer": acute_results["metadata"]["link"],
                    "status_code": 203
                }
                print(external_api_data)
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
        except Exception as e:
            error_message = f"급성 질문 처리 중 오류 발생: {str(e)}"
            print(error_message)
            external_api_data = {
                "sessionId": session_id,
                "uid": uid,
                "answer": error_message,
                "status_code": 500
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

    if request.isAcute:
        background_tasks.add_task(process_acute_question, request.sessionId, request.uid, request.question)
        return {"message": "응답이 성공적으로 처리되었습니다."}

    async def process_question(session_id: str, uid: str, question: str, info: str):
        try:
            # 질문과 관련된 상위 5개 문서 검색
            top_docs = db.search(query=question, k=5)

            # 관련 문서 내용 추출
            context = ""
            for doc in top_docs:
                if doc['metadata']['contextualized_content'] == "":
                    context += f"{doc['metadata']['original_content']}\n\n"
                else:
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

Information and interests of the elderly person who asked the question.:
<user_information>
{info}
</user_information>

Please follow these steps:

1. Carefully read and analyze the document.
2. Identify the key information in the document that relates to the question.
3. Formulate a concise answer that directly addresses the question.
4. Ensure your answer is simple and easy for an elderly person to understand.
5. If the document doesn't contain information to fully answer the question, state this clearly and provide whatever relevant information you can from the document in "plain_text" poster template format. In other cases, please use a "qna__square_single" poster template.
6. IMPORTANT: If the question is CLOSELY RELATED to any of the following topics, just state through "plain_text" that the content cannot be found in the document. "문서에는 해당 질문과 관련된 내용이 보이지 않습니다. 다른 질문을 해주시면 성심껏 답변드리겠습니다." is all you need to say.: [{', '.join(DESIRED_TERMS)}]

When writing your response:
- Use simple, clear language
- Avoid complex terms or jargon
- Keep sentences short and to the point
- Be respectful and patient in your tone
- Be concise and to the point
- Use Markdown to **bold** important parts of your answer

Please provide your answer in Korean, formatted as plain text without any special formattings. Your response should be concise, typically no more than 3-4 sentences, unless more detail is absolutely necessary to answer the question fully.

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
                                    "enum": ["plain_text", "qna__square_single"]
                                },
                                "content": {
                                    "anyOf": [
                                        {
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
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Plain text content for the template."
                                                }
                                            },
                                            "required": ["text"],
                                            "additionalProperties": False
                                        }
                                    ]
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
            
            # Parse the raw response
            response_data = json.loads(raw_response)
            
            # Prepare external API data based on template type
            if response_data["template_type"] == "plain_text":
                print("Plain text template")
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "answer": response_data["content"]["text"],
                    "status_code": 201
                }
            else:
                print("QnA template")
                external_api_data = {
                    "sessionId": session_id,
                    "uid": uid,
                    "answer": raw_response,
                    "status_code": 202
                }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
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
                async with aiohttp.ClientSession() as session:
                    async with session.post(EXTERNAL_API_URL, json=external_api_data) as response:
                        print(await response.text())
            except Exception as e:
                print(f"외부 API 호출 중 오류 발생: {str(e)}")

    # 비동기 작업 시작
    # # Add the background task and return response
    background_tasks.add_task(process_question, request.sessionId, request.uid, request.question, request.info)
    return {"message": "응답이 성공적으로 처리되었습니다."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5056)
