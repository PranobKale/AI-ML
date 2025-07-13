from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2bit",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class UserInfo(BaseModel):
    name: str = Field(description='name of the person')
    age: int = Field(gt=18,description='age of the person')
    city: str = Field(description='city of the person')

parser = PydanticOutputParser(pydantic_object=UserInfo)

template = PromptTemplate(
    template='give the name,city and age of the person in the following {palce} \n {format_instruction}',
    input_variables=['place'],
    partial_variables = {'format_instruction':parser.get()}
)

# prompt = template.invoke({'place':'India'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# chain 
chain =  template | model | parser

final_result = chain.invoke({'place':'india'})

print(final_result)
