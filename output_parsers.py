
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()

# 1. Output in comma separated structure
print('1. Output in comma separated structure\n')
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print('Format instructions:')
print(format_instructions)

prompt = PromptTemplate(
    template="Provide 5 examples of {query}.\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

llm = OpenAI(temperature=.9, model="text-davinci-003")
prompt = prompt.format(query="Currencies")
print(prompt)
output = llm(prompt)
print(output)


# 2. Output in JSON structure
print('\n2. Output in JSON structure')
response_schemas = [
    ResponseSchema(name="currency", description="answer to the user's question"),
    ResponseSchema(name="abbrevation", description="Whats the abbrebation of that currency")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
print('Output parser:')
print(output_parser)

format_instructions = output_parser.get_format_instructions()
print('Format instructions:')
print(format_instructions)

prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

prompt = prompt.format(query="what's the currency of America?")
print('Prompt:')
print(prompt)

output = llm(prompt)
print(output)
