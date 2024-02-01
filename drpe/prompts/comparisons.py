# Prompt templates for comparison of two NLG models geneartions


COMPARISON_PROMPT = """Read this News: {text}

There are 2 summaries of the above article:
1. {summary_1}
2. {summary_2}"""

FEW_SHOT_PROMPT = """Assuming you are {role_type} <{role_description}>, please select a better summary to above article in your point of view from above two candidates:"""
