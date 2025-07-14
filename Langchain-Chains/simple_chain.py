from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser


load_dotenv()

prompt = PromptTemplate(
    template= 'generate some interesting facts aout the {topic}',
    input_variables= ['topic']
)

model = ChatOpenAI()

# prompt = template.invoke({'topic': 'Black Hole'})

parser = StructuredOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'Black hole'})

print(result)

# to visualise the template
chain.get_graph().print_ascii()
