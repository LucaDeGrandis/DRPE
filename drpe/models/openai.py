from langchain.chains import LLMChain
from langchain_openai import OpenAI
from drpe.models.llm import Llm


class LlmOpenAI(Llm):
    def __init__(self, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, **kwargs):
        self.model = OpenAI(
            model=kwargs['model'],
            temperature=kwargs['temperature'],
            openai_api_key=kwargs['openai_api_key'],
            **{'seed': kwargs['seed']}
        )

    def __call__(self, input, **kwargs):
        chain = LLMChain(llm=self.model, prompt=self.prompt)
        return chain.invoke(input=input)
