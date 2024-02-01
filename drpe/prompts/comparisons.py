# Prompt templates for comparison of two NLG models geneartions


COMPARISON_PROMPT = """Read this News: {text}
"""

FEW_SHOT_PROMPT = """Assuming you are {role} <{description}>, please select a better summary to above article in your point of view from above two candidates:"""
