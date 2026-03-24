import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)

response = client.responses.create(
    model="llama-3.1-8b-instant",
    input="""
    extraia informações do evento: Daniel e Alberto estão organizando um evento de tecnologia no dia 15 de julho de 2024, às 19h, no auditório da empresa. 
    O evento contará com palestras sobre inteligência artificial e networking entre os participantes. 
    O objetivo é promover a troca de conhecimentos e experiências na área de tecnologia.
    EXEMPO DA FORMATAÇÃO:
    {
      "formato_saida":{
        "pessoas":["..."],
        "ação":"...",
        "tipo_evento":"...",
        "data":"...", 
      }
    }
    """,
)

print(response.output_text)
