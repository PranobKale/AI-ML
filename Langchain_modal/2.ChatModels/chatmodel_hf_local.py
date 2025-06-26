from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# note :- this will donwload the whole llm on the system and  then run and provide the o/p.
llm = HuggingFacePipeline(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', # need to provide the repo name in terms of id to find out the which model we are using
    task='text-generation', # need to provide the task name,
    pipeline_kwargs=dict(
        temprature = 0.5,
        max_new_tokens = 100
    ) # we can mention requrired parameters as well
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the cpaital of India")

print(result.content)