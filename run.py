from typing import List, Dict, Any, Tuple
import argparse
import json
import os

from sentence_transformers import SentenceTransformer

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI

from drpe.prompts.dynamic_roles import DYNAMIC_TEMPLATES
from drpe.prompts.comparisons import COMPARISON_PROMPT, FEW_SHOT_PROMPT, SUFFIX_PROMPT

from sklearn.cluster import KMeans
import numpy as np


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
) -> Tuple[str, str]:
    """ Generates the dynamic roles

    Args:
        model: The model used to generate the dynamic roles.
        input_text: The input text.

    Returns:
        A tuple containing the coarse grained roles and the fine grained roles.
        The tuple has two elements since the output is unpocessed text from the LLM.
    """
    # Generate coarse grained roles
    coarse_grained_prompt = PromptTemplate.from_template(DYNAMIC_TEMPLATES['coarse_grained'])
    chain = LLMChain(llm=model, prompt=coarse_grained_prompt)
    coarse_grained_roles = chain.invoke(input=input_text)['text']

    # Generate fine grained roles
    fine_grained_prompt = PromptTemplate.from_template(DYNAMIC_TEMPLATES['fine_grained'])
    chain = LLMChain(llm=model, prompt=fine_grained_prompt)
    fine_grained_roles = chain.invoke(input=input_text)['text']

    return coarse_grained_roles, fine_grained_roles


def distance_measure(
    vector_a,
    vector_b
):
    return sum([(a - b) ** 2 for a, b in zip(vector_a, vector_b)])


def dynamic_roles_clutering(
    generated_roles: List[List[str]],
    model_name: str,
    n_clusters: int
):
    """ Clusters text using embeddings and k-means """
    # Create the embeddings
    embedding_model = SentenceTransformer(model_name)
    roles_concatenated = [': '.join(x) for x in generated_roles]
    if 'uncased' in model_name.lower():
        sentence_embeddings = embedding_model.encode([el.lower() for el in roles_concatenated])
    else:
        sentence_embeddings = embedding_model.encode(roles_concatenated)

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(sentence_embeddings)

    # Find the elements closest to the centroids
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    roles = []
    for i in range(n_clusters):
        cluster_elements = list(filter(lambda x: x[2] == i, zip(generated_roles, sentence_embeddings, labels)))
        distances = [distance_measure(centers[i], el[1]) for el in cluster_elements]
        closest_index = np.argmin(distances)
        roles.append(cluster_elements[closest_index][0])

    return roles


def dynamic_role_parser(
    text: str
) -> Dict[str, str]:
    """ Parses the output of LLM to extract the dynamic roles

    Args:
        text: The output of LLM.

    Returns:
        A list of roles.

    Warning:
        This function is not robust to changes in the output of LLM.
        For as it is right now the LLM is expected to follow the following format:
            1. role 1: role description
            2. role 2: role description
            ...
    """
    # parse the roles
    roles_raw = [x.strip() for x in text.split('\n')]
    roles_raw = list(filter(lambda x: x, roles_raw))
    roles = [x[3:].strip() for x in roles_raw]

    # Save the role types and descriptions
    types_descriptions = []
    for role in roles:
        role_split = role.split(':')
        types_descriptions.append([
            role_split[0].strip(),
            ':'.join(role_split[1:])
        ])

    return roles


def evaluator(
    model,
    input_document: str,
    summaries: List[str],
    roles: List[List[str]],
):
    """ Runs the evaluation of two summaries using the dynamic roles """
    # Define the formatted roles for the few shot prompt template
    examples = []
    for line in roles:
        examples.append({
            'role_type': line.split(':')[0].strip(),
            'role_description': line.split(':')[1].strip(),
        })

    example_prompt = PromptTemplate(
        input_variables=["role_type", "role_description"], template=FEW_SHOT_PROMPT
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=COMPARISON_PROMPT,
        input_variables=['text', 'summary_1', 'summary_2'],
        suffix=SUFFIX_PROMPT,
    )

    # Evaluate the summaries
    chain = LLMChain(llm=model, prompt=prompt)

    return chain.invoke(input={
        'text': input_document.replace('\n', ' ').replace('\t', ' ').replace('  ', ' '),
        'summary_1': summaries[0],
        'summary_2': summaries[1]
    })


def argparser():
    parser = argparse.ArgumentParser(description='Run the program')
    parser.add_argument('dataset', type=str, help='The datset path. It must be a jsonl file.')
    parser.add_argument('out_file', type=str, help='The output path. It must be a jsonl file.')
    parser.add_argument('openai_key', type=str, help='The OpenAI key.')
    parser.add_argument('--verbose', action='store_true', help='Saves intermediate results.')
    parser.add_argument('--roles_generator', type=str, default='gpt-3.5-turbo-instruct', help='The model used to generate the dynamic roles.')
    parser.add_argument('--roles_generator_templates', type=int, default=32, help='The model used to generate the dynamic roles.')
    parser.add_argument('--embedding_gnerator', type=str, default='all-MiniLM-L6-v2', help='The model used to generate embeddings for roles clustering.')
    parser.add_argument('--roles_clusters', type=int, default=4, help='Number of dynamic roles.')
    parser.add_argument('--evaluator', type=str, default='gpt-3.5-turbo-instruct', help='The model used to evaluate the summaries.')
    return parser.parse_args()


def __main__():
    args = argparser()

    # Load the datset
    dataset = load_jsonl_file(args.dataset)

    # Create the model
    roles_generator = OpenAI(model=args.roles_generator, temperature=0, openai_api_key=args.openai_key, model_kwargs={'seed': 42})
    evaluation_model = OpenAI(model=args.evaluator, temperature=0, openai_api_key=args.openai_key, model_kwargs={'seed': 42})

    results = []

    for _el in dataset:
        res = {}

        # Generate the roles
        coarse_grained_roles, fine_grained_roles = dynamic_roles_generator(roles_generator, _el['text'])
        roles = dynamic_role_parser(coarse_grained_roles) + dynamic_role_parser(fine_grained_roles)
        roles_clustered = dynamic_roles_clutering(roles, args.embedding_gnerator, args.roles_clusters)
        if args.verbose:
            res['generated_roles'] = {
                'coarse_grained_roles': coarse_grained_roles,
                'fine_grained_roles': fine_grained_roles
            }
            res['clustered_roles'] = roles_clustered
            results.append(res)

        # Run the evaluation
        evaluation = evaluator(
            evaluation_model,
            _el['text'],
            _el['summaries'],
            roles_clustered,
        )
        res['evaluation'] = evaluation['text']
        break

    write_jsonl_file(args.out_file, results, overwrite=True)


if __name__ == '__main__':
    __main__()
