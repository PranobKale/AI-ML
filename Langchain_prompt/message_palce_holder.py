from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate

# chat template
chat_tempalate = ChatPromptTemplate([
    ('system','you are helpfull AI agnet'),
    MessagesPlaceholder(variable_name='chat_history')
    ('human','{query}')
])

# load a chat history
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# create a final prompt
prompt = chat_tempalate.invoke({'chat_history':chat_history,'query':'Where is my refund'})

print(prompt)