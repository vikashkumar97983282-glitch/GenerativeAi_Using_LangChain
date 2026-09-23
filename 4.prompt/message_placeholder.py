from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# chat template 
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# load chat history
with open('prompt/chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'I have an issue with my order. Can you help me?'})

print(prompt)