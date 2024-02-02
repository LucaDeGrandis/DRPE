from langchain_openai import ChatOpenAI
from drpe.models.llm import Llm


class LlmChatOpenAI(Llm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self, **kwargs):
        self.model = ChatOpenAI(
            model=kwargs['model'],
            temperature=kwargs['temperature'],
            openai_api_key=kwargs['openai_api_key'],
            **{'seed': kwargs['seed']}
        )

    def __call__(self, input, **kwargs):
        chain = self.prompt | self.model
        return chain.invoke(input=input)
