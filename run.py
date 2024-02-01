from langchain.chains import LLMChain
from langchain_openai import OpenAI
from typing import List, Dict, Any
import argparse
import json
import os


from drpe.prompts.dynamic_roles import DYNAMIC_TEMPLATES


def load_jsonl_file(
    filepath: str
) -> List[Dict[str, Any]]:
    """ Load a jsonl file into a list """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str = 'a+',
    overwrite: bool = False
) -> None:
    """ Write a list into a jsonl """
    if overwrite:
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')


def dynamic_roles_generator(
    model,
    input_text: str,
):
    # Generate coarse grained roles
    coarse_grained_prompt = DYNAMIC_TEMPLATES['coarse_grained']
    chain = LLMChain(llm=model, prompt=coarse_grained_prompt)
    coarse_grained_roles = chain.invoke(text=input_text)['text']

    # Generate fine grained roles
    fine_grained_prompt = DYNAMIC_TEMPLATES['fine_grained']
    chain = LLMChain(llm=model, prompt=fine_grained_prompt)
    fine_grained_roles = chain.invoke(text=input_text)['text']

    return coarse_grained_roles, fine_grained_roles


def argparser():
    parser = argparse.ArgumentParser(description='Run the program')
    parser.add_argument('dataset', type=str, help='The datset path. It must be a jsonl file.')
    parser.add_argument('out_file', type=str, help='The output path. It must be a jsonl file.')
    parser.add_argument('--roles_generator', type=str, default='gpt-3.5-turbo-1106', help='THe model used to generate the dynamic roles.')
    parser.add_argument('--roles_generator_templates', type=int, default=32, help='THe model used to generate the dynamic roles.')


def __main__():
    args = argparser()

    # Load the datset
    dataset = load_jsonl_file(args.dataset)

    # Create the model
    model = OpenAI(model=args.roles_generator, temperature=0, model_kwargs={'seed': 42})

    for _el in dataset:
        # Generate the roles
        coarse_grained_roles, fine_grained_roles = dynamic_roles_generator(model, _el['text'])
        break
