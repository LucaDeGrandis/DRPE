# Prompts for the generation of dynamic roles


DYNAMIC_COARSE_GRAINED_TEMPLATE = """Read this News: {text}

Please categorize several types of users with general intent for this news:
"""


DYNAMIC_FINE_GRAINED_TEMPLATE = """Read this News: {text}

Please categorize several types of users and consider the extent to which people know about events mentioned in above article:
"""


DYNAMIC_TEMPLATES = {
    'coarse_grained': DYNAMIC_COARSE_GRAINED_TEMPLATE,
    'fine_grained': DYNAMIC_FINE_GRAINED_TEMPLATE,
}
