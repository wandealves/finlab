import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    input="extraia informações do evento: Daniel e Alberto estão organizando um evento de tecnologia no dia 15 de julho de 2024, às 19h, no auditório da empresa.O evento contará com palestras sobre inteligência artificial e networking entre os participantes.O objetivo é promover a troca de conhecimentos e experiências na área de tecnologia.",
    text_format=CalendarEvent,
)

event = response.output_parsed
print(event.model_dump_json(indent=2))
