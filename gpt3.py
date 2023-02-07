import os
import json
import requests
import openai
from prompt import prompt, stops
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('LM_OPENAI_API_KEY')


def complete(prompt, 
             stops=None, 
             max_tokens=150, 
             temperature=0.9, 
             engine='text-davinci-003'):

    response = openai.Completion.create(
        engine=engine, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        temperature=temperature,
        stop=stops)
    
    completion = response.choices[0].text
    response = completion.strip()

    return response


def get_gpt3_answer(question):
    input_text = prompt+f'Question: {question}\nAnswer:'
    completion = complete(
        input_text, 
        stops,
        max_tokens=200,
        temperature=0.9)
    return completion
