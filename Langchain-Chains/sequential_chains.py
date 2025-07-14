from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
    template="generate a detailed report on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Extract the 5 important pont out of following textr \n {text}',
    input_variables=['text']
)
model = ChatOpenAI()

parser = StructuredOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"unemployment in the india"})

chain.get_graph().draw_ascii()