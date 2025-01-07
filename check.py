
import os
from dotenv import load_dotenv

def load():
    load_dotenv()
    print("OPENAI_API_KEY:", os.getenv('OPEN_AI_API'))

load()
