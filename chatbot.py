from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
template = '''
You are a helpful assistant that translates that answers in a concise manner.
{input}

here is conversation history:
{history}   
Answer : 
'''
client = ChatOllama(model="llama3.2") 
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | client

result = chain.invoke({"history":"" , "input":"What is the capital of France?"})
print(result.content)