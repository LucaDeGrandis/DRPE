from abc import ABC, abstractmethod


class Llm(ABC):
    def __init__(self, **kwargs):
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs):
        pass
    
    def add_prompt(self, prompt):
        self.prompt = prompt

    @abstractmethod
    def __call__(self, text, **kwargs):
        pass
