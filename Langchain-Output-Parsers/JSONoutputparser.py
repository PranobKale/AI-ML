from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2bit",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Give me the name ,age and city of the fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables= {'format_instruction':parser.get_format_instructions()}
)

# prompt = template1.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# with the chain

chain = template1 | model | parser

final_result = chain.invoke({})

print(final_result)





# template2 = PromptTemplate(
#     template='write a 5 line summary on the following text. /n {text}',
#     input_variables = ['text']
# )

