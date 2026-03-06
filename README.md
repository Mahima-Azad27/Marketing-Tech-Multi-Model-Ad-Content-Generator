# Marketing-Tech-Multi-Model-Ad-Content-Generator
from transformers import pipeline
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, HF_MODEL


class MultiModelManager:

    def __init__(self):

        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        self.hf_generator = pipeline(
            "text2text-generation",
            model=HF_MODEL
        )

    def generate_openai(self, prompt):

        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def generate_huggingface(self, prompt):

        result = self.hf_generator(prompt, max_length=200)

        return result[0]["generated_text"]
