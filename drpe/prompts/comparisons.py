# Prompt templates for comparison of two NLG models geneartions


# COMPARISON_PROMPT = """Read this News: {text}

# There are 2 summaries of the above article:
# 1. {summary_1}
# 2. {summary_2}"""

# FEW_SHOT_PROMPT = """Assuming you are {role_type} <{role_description}>, please select a better summary to above article in your point of view from above two candidates:"""


COMPARISON_PROMPT = """Read this News: {text}

There are 2 summaries of the above article:
1. {summary_1}
2. {summary_2}

Here is a list of roles and their descriptions:
"""

FEW_SHOT_PROMPT = """- {role_type} <{role_description}>"""


SUFFIX_PROMPT = """Do the following:
1. For each of the roles described above, select the summary that best fits the role and the reason.
2. Forma the output as a list of dictionaries with keys "preferred_summary" and "reason". The preferred_summary should be the index of the preferred summary (wither 1 or 2).

Remember, the output list should have one element for each role described in above.

JSON:"""
