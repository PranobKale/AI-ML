from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# dynamic prompt if need to send multiple chats
chat_tempalate = ChatPromptTemplate([
    ('system','you are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')
])

prompt = chat_tempalate.invoke({'domain':'Cricket','topic':'Dusra'})

print(prompt)