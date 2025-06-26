from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embbeding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    'Delhi is the capital of india',
    'my name is pranob kale',
    'kolcatta is the capital of west bengal',
    'paris is the capital of france'
]

result = embbeding.embed_documents(documents)

print(result)