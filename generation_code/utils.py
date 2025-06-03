import re
import random

SYSTEM_PROMPTS_XSUM=[
    "You are a journalist from the United Kingdom writing for a national newspaper on a broad range of topics",
]

SYSTEM_PROMPT_M4ABS=[
    "You are a university professor working in the academic field",
]

USER_PROMPTS_XSUM_INFORMED=[
    "Write a piece of news, that will appear in a national newspapers in the UK and that has the following title:\n\n{m}\nIn writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 500 words.",
]

USER_PROMPTS_M4ABS_INFORMED=[
    "Write an abstract for a scientific paper that has the following title:\n\n{m}\n\nDon't use any formatting and do not repeat the title and use at most 500 words."
]

def get_prompt_m4abs(m, informed, use_system_prompt=True):
    if use_system_prompt:
        out = [{"role":"system", "content":random.choice(SYSTEM_PROMPT_M4ABS)},]
    else:
        out = []
    if not informed:
        raise NotImplementedError
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_M4ABS_INFORMED).format(m=m)})
    return out

def get_random_prompt_xsum(m, informed, use_system_prompt=True):
    if use_system_prompt:
        out = [{"role":"system", "content":random.choice(SYSTEM_PROMPTS_XSUM)},]
    else:
        out = []
    if not informed:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM).format(m=m)})
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM_INFORMED).format(m=m)})
    return out


USER_PROMPTS_XSUM_LINGUISTIC_INFORMED=[
    "Write a piece of news, that will appear in a national newspapers in the UK and that has the following title:\n\n{m}\n"
    "In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague."
    "You don't have to add the date of the event but you can, use at most 500 words."
    "Please also adherethe to the following stylistic guidelines:\n"
    " - use short sentences and use long words to improve the writing quality;\n" # align sentence_length and char_per_tok
    " - don't refrain from using adjectives and adverbs;\n" # increase align lexical_density and ttr_...
    " - limit the use of determiners such as \"this\", \"which\" and \"any\";\n" # lower upos_dist_DET
    " - feel free to use the verb before the subject.", # increase subj_post
]

def get_random_prompt_xsum_linguistic(m, informed):
    out = [{"role":"system", "content":random.choice(SYSTEM_PROMPTS_XSUM)},]
    if not informed:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM_LINGUISTIC).format(m=m)})
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM_LINGUISTIC_INFORMED).format(m=m)})
    return out


def postprocess_text(text):
    try:
        text = re.sub(r"\s+", " ", text)
        text = text.lstrip("#* ")
    except:
        pass
    
    return text