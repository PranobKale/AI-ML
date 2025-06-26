from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', # need to provide the repo name in terms of id to find out the which model we are using
    task='text-generation' # need to provide the task name
)

model = ChatHuggingFace(llm=llm)

retsult = model.invoke("what is the capital if India")

print(retsult.content)
