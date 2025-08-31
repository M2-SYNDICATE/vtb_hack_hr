from pydantic import BaseModel
from typing import List, Optional

# --- Для вакансии ---
class JobPosting(BaseModel):
    comment: str   # описание вакансии или дополнительные инструкции
    answer: bool   # можно использовать, если нужно подтверждение для проверки

# --- Новый класс для вопроса с примером ответа ---
class QuestionWithAnswer(BaseModel):
    question: str
    example_answer: Optional[str]  # сюда LLM будет подставлять пример ответа

# --- Блоки вопросов ---
class GeneralQuestions(BaseModel):
    questions: List[QuestionWithAnswer]

class ProfessionalQuestions(BaseModel):
    questions: List[QuestionWithAnswer]

class ExperienceQuestions(BaseModel):
    questions: List[QuestionWithAnswer]

class SituationalQuestions(BaseModel):
    questions: List[QuestionWithAnswer]

class GrowthQuestions(BaseModel):
    questions: List[QuestionWithAnswer]

# --- Общий формат интервью ---
class InterviewQuestions(BaseModel):
    general: GeneralQuestions
    professional: ProfessionalQuestions
    experience: ExperienceQuestions
    situational: SituationalQuestions
    growth: GrowthQuestions