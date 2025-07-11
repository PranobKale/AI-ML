from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embbeding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embbeding.embed_query('Delhi is the capital of India')

print(result)