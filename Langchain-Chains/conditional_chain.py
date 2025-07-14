from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    semtiment: Literal['positive','negative'] = Field(description='Prvide the semtiment for te feeback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Provide a sentiment as positive or nigative from the feedback provided \n {feedback}',
    input_variables=['feedback'],
    partial_variables=[{'format_instruction':parser2.get_format_instructions()}]
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="priovide the positive response on the following \n  {feedback}",
    input_variables=['feeback']
)

prompt3 = PromptTemplate(
    template="priovide the positive response on the following \n  {feedback}",
    input_variables=['feeback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'nigative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is a beautiful phone.'})

