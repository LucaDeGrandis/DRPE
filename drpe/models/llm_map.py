from drpe.models.openai_chat import LlmChatOpenAI
from drpe.models.openai import LlmOpenAI


MODELS = {
    'gpt-3.5-turbo': LlmChatOpenAI,
    'gpt-3.5-turbo-1106': LlmChatOpenAI,
    'gpt-3.5-turbo-instruct': LlmOpenAI,
}
