from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from module1.convert_functions import *
from pydantic import BaseModel, Field
from typing import List, Optional
import translitua
import re  # <-- ДОБАВЛЕНО: Импорт модуля для регулярных выражений

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Не найден API ключ OPENROUTER_API_KEY")


class Question(BaseModel):
    question: str = Field(description="Текст вопроса для собеседования.")
    expected_response: Optional[str] = Field(
        description="Возможный ожидаемый ответ кандидата на основе CV."
        )


class InterviewQuestions(BaseModel):
    general_questions: List[Question] = Field(
        description="Общие вопросы: опыт работы, зарплатные ожидания, удобство добирания до работы и подобные."
        )
    hard_skills_questions: List[Question] = Field(
        description="Вопросы на проверку hard skills (технические навыки), основанные на вакансии и CV."
        )
    soft_skills_questions: List[Question] = Field(
        description="Вопросы на проверку soft skills (поведенческие, ситуационные навыки)."
        )


# --- Ваша логика транслитерации (без изменений) ---
custom_translit_map = {
    "Excel": "Эксель",
    "Word": "Ворд",
    "Visio": "Визио",
    "BIOS": "БИОС",
    "BMC": "Би-Эм-Си",  # Произношение по буквам предпочтительнее для TTS
    "RAID": "Рейд",
    "CMDB": "Си-Эм-Ди-Би",
    "DCIM": "Ди-Си-Ай-Эм",
    "IPMI": "Ай-Пи-Эм-Ай",
    "LAN": "Лан",
    "SAN": "Сан",
    }


def transliterate_word(word):
    # Сначала ищем в ручном словаре (без учета регистра)
    for k, v in custom_translit_map.items():
        if k.lower() == word.lower():
            return v
    # Если не нашли, используем библиотеку
    return translitua.translit(word)


def process_question_for_tts(question_text):
    # Находим все слова на латинице
    latin_words = re.findall(r'[a-zA-Z-]{2,}', question_text)

    processed_text = question_text
    # Заменяем каждое найденное слово на его транслитерацию
    for word in set(latin_words):  # set() чтобы не заменять одно и то же слово несколько раз
        transliterated = transliterate_word(word)
        # Используем re.sub с \b для замены только целых слов
        processed_text = re.sub(r'\b' + re.escape(word) + r'\b', transliterated, processed_text)

    return processed_text


# --- Конец блока вашей логики ---

def prompt_question_block(info: dict, cv_text: str) -> List[dict]:
    """
    Функция для создания промпта для генерации блока вопросов.
    """
    prompt_content = f"""
    Описание вакансии: {info}
    Текст резюме кандидата: {cv_text}

    Сгенерируй около 15 вопросов для первичного HR-собеседования в общей сложности.
    Раздели вопросы на три категории:
    - general_questions: Общие вопросы, такие как опыт работы, зарплатные ожидания, удобно ли добираться до работы. Должно быть 3-5 вопросов.
    - hard_skills_questions: Вопросы на проверку hard skills (технические навыки). Основывайся на требованиях вакансии и опыте из CV.
    - soft_skills_questions: Вопросы на проверку soft skills (коммуникация, командная работа, решение проблем и т.д.).

    Регулируй количество вопросов в зависимости от типа вакансии:
    - Если вакансия техническая (IT, инженерия, разработка и т.п.), сделай больше вопросов по hard skills (7-10) и меньше по soft skills (3-5).
    - Если вакансия больше ориентирована на soft skills (менеджмент, продажи, HR и т.п.), сделай больше вопросов по soft skills (7-10) и меньше по hard skills (3-5).
    - Для смешанных вакансий распредели примерно поровну.

    expected_response ты заполняешь только для hard_skills и general_questions. В soft_skills ты оставляешь его null

    Общее количество вопросов должно быть около 15, включая общие.
    НИ В КОЕМ СЛУЧАЕ НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЕ СИМВОЛЫ ИЛИ ТЕРМИНЫ, ВСЕ ПЕРЕВОДИ НА РУССКИЙ, ДАЖЕ НАЗВАНИЯ СОФТА ИЛИ ЕЩЕ ЧЕГО ЛИБО, ТОЛЬКО РУССКИЙ ЯЗЫК
    """
    return [
        {"role": "system",
         "content": "Ты — HR-ассистент, генерирующий вопросы для собеседования на основе вакансии и CV."},
        {"role": "user", "content": prompt_content}
        ]


def question_block(info_cv: str, json_path: str) -> dict:
    """
    Function for processing the JSON with CV analyses, generating question blocks only for candidates with "answer": true.
    For each such candidate, reads the full CV text from the path (key in JSON), generates questions, and returns a separate dict with questions.
    :param info_cv: path to info about job
    :param json_path: path to the JSON file with CV analyses (invo_cv_text)
    :return: returns a dict with CV paths as keys and their corresponding questions (or None if "answer": false)
    """
    info, _ = convert_to_dict(info_cv)

    with open(json_path, 'r', encoding='utf-8') as f:
        invo_cv = json.load(f)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
        )

    result = {}
    for cv_path, data in invo_cv.items():
        if data.get("answer", False):
            cv_text = convert_to_text(cv_path, file_num=0)

            response = client.chat.completions.parse(
                model="deepseek/deepseek-r1-0528:free",
                messages=prompt_question_block(info=info, cv_text=cv_text),
                response_format=InterviewQuestions,
                temperature=0.1,
                top_p=0.95
                )

            questions = response.choices[0].message.parsed

            # --- НАЧАЛО ВНЕДРЕННОГО БЛОКА ---
            # Пост-обработка сгенерированных вопросов для TTS.
            # Несмотря на инструкцию в промпте, модель может иногда использовать латиницу.
            # Этот блок гарантирует, что все термины будут транслитерированы.

            for q_list in [questions.general_questions, questions.hard_skills_questions,
                           questions.soft_skills_questions]:
                for item in q_list:
                    item.question = process_question_for_tts(item.question)
            # --- КОНЕЦ ВНЕДРЕННОГО БЛОКА ---

            result[cv_path] = questions.model_dump()

        else:
            result[cv_path] = None

    return result


# Пример использования
print(question_block(info_cv=r"D:\download\Описание ИТ.docx",
                     json_path=r"D:\pycharm\vtb_hack_hh\module1\results\cv_validation_results_20250907_140144.json"))