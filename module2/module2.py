from openai import OpenAI
from dotenv import load_dotenv
import os
from module1.convert_functions import *
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Не найден API ключ OPENROUTER_API_KEY")

def question_block(info_cv: str, info_file: str) -> dict:
    """
    Function for generating question block with 5 categories:
    (general, professional, experience, situational, growth)
    every question contains what candidate can say in response
    :param info_cv: path to info about job
    :param info_file: path to candidate with True value
    :return: returns dict with answers dict[dict[question,answer]
    """
    info = convert_to_dict(info_cv)
    info_cv = convert_to_text(info_file,file_num=0)
    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
            )
    response = client.chat.completions.parse(
            model="openai/gpt-oss-20b:free",
            messages=prompt_question_block(info=info,cv_text=info_cv),
            response_format=InterviewQuestions,
            temperature=0.3,
            top_p=0.95
            )
    result = response.choices[0].message.parsed
    return result
