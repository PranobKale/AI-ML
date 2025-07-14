from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 questions and answers from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = "In the rapidly evolving digital economy, businesses must adapt to emerging technologies to remain competitive. Artificial intelligence, blockchain, and cloud computing have redefined operational efficiency and customer engagement. Companies leveraging these tools are seeing improved decision-making, reduced costs, and scalable growth. However, digital transformation is not without challenges—security risks, data privacy concerns, and the need for upskilling employees remain major hurdles.Successful adoption requires a well-planned strategy, leadership support, and continuous innovation. Organizations must foster a culture of learning and agility to navigate disruptions. As technology continues to advance, those who embrace change and invest in smart digital infrastructure will be the leaders of tomorrow’s economy."

result = chain.invoke({'text':text})

print(result)