
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
# load_dotenv() is a function that loads variables from a .env file into environment variables in a Python script.
# It allows you to store sensitive information or configuration settings separate from your code
# and access them within your application.
from dotenv import load_dotenv
load_dotenv()


llm = OpenAI(model_name="text-davinci-003")
template = """
{our_text}

Can you create a post for tweet in {wordsCount} words for the above?
"""
prompt = PromptTemplate(
    input_variables=["wordsCount", "our_text"],
    template=template,
)
final_prompt = prompt.format(wordsCount='3',
                             our_text="I love trips, and I have been to 6 countries. I plan to visit few more soon.")

print(final_prompt)
print(llm(final_prompt))
