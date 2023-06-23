
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv
load_dotenv()

# 1. Basic prompt with no example selectors
our_prompt = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Question: What is a house?
Response: """
llm = OpenAI(temperature=.9, model="text-davinci-003")

print('\n1. Basic prompt with no example selectors')
print(our_prompt)
print(llm(our_prompt))


# 2. Few shot example with a few example selectors
examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }
]
example_template = """
Question: {query}
Response: {answer}
"""
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
prefix = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Here are some examples: 
"""
suffix = """
Question: {userInput}
Response: """
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)
query = "What is a house?"

print('\n2. Few shot example with a few example selectors')
print(few_shot_prompt_template.format(userInput=query))
print(llm(few_shot_prompt_template.format(userInput=query)))


# 3. Few shot example with more example selectors and length-based example selector
# It offers features such as the ability to include or exclude examples based on the length of the query.
# This is important because there is a maximum context window limitation for prompt and generation output length.
# The goal is to provide as many examples as possible for few-shot learning without exceeding the context window
# or increasing processing times excessively.
# The dynamic inclusion/exclusion of examples means that we choose which examples to use based on certain rules.
# This helps us use the model's abilities in the best way possible.
# It allows us to be efficient and make the most out of the few-shot learning process.
examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }, {
        "query": " What are your ambitions?",
        "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!"
    }, {
        "query": "What happens when you get sick?",
        "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!"
    }, {
        "query": "WHow much do you love your dad?",
        "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!"
    }, {
        "query": "Tell me about your friend?",
        "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!"
    }, {
        "query": "What math means to you?",
        "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!"
    }, {
        "query": "What is your fear?",
        "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!"
    }
]
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=200
)
new_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)
query = "What is a house?"
print('\n3. Few shot with more example selectors and length-based example selector')
print(new_prompt_template.format(userInput=query))
print(llm(new_prompt_template.format(userInput=query)))
