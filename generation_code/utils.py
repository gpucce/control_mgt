import random

SYSTEM_PROMPTS_DICE=[
    "You are an Italian journalist writing for a national newspaper focusing on criminal events happening in the area surrounding Modena",
    "You are an Italian-French journalist writing in Italian about criminal events happening in the area surrounding Modena",
    "You are an Italian journalist writing for a local newspaper focusing on criminal events happening in the area surrounding Modena",
    "You are an Italian-American journalist writing for a local newspaper focusing on criminal events happening in the area surrounding Modena",
]

USER_PROMPTS_DICE=[
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}",
    "Write a piece of news in Italian, that will appear in a national Italian newspaper and that has the following title:\n\n{m}",
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}",
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}",
]

USER_PROMPTS_DICE_INFORMED=[
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 300 words.",
    "Write a piece of news in Italian, that will appear in a national Italian newspaper and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 300 words.",
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 300 words.",
    "Write a piece of news in Italian, that will appear in a local Italian newspaper and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 300 words.",
]

def get_random_prompt_dice(m, informed):
    out = [{"role":"system", "content":random.choice(SYSTEM_PROMPTS_DICE)},]
    if not informed:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_DICE).format(m=m)})
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_DICE_INFORMED).format(m=m)})
    return out

SYSTEM_PROMPTS_DICE_ANITA=[
    "Sei un giornalista italiano che che scrive per un giornale nazionale focalizzandosi su eventi criminali che accadono a Modena",
    "Sei un giornalista italo-francese che scrive in italiano su eventi criminali che accadono a Modena",
    "Sei un giornalista italiano che scrive per un giornale locale focalizzandosi su eventi criminali che accadono a Modena",
    "Sei un giornalista italo-americano che scrive per un giornale locale focalizzandosi su eventi criminali che accadono a Modena",
]

USER_PROMPTS_DICE_ANITA=[
    "Scrivi un articolo di giornale in italiano. L'articolo sarà pubblicato su un giornale locale e avrà il seguente titolo:\n\n{m}",
    "Scrivi un articolo di giornale in italiano. L'articolo sarà pubblicato su un giornale nazionale e avrà il seguente titolo:\n\n{m}",
    "Scrivi un articolo di giornale in italiano. L'articolo sarà pubblicato su un giornale locale e avrà il seguente titolo:\n\n{m}",
    "Scrivi un articolo di giornale in italiano. L'articolo sarà pubblicato su un giornale locale e avrà il seguente titolo:\n\n{m}",
]

def get_random_prompt_dice_anita(m):
    return [
        {"role":"system", "content":random.choice(SYSTEM_PROMPTS_DICE_ANITA)},
        {"role":"user", "content":random.choice(USER_PROMPTS_DICE_ANITA).format(m=m)}
    ]

SYSTEM_PROMPTS_CItA = [
    "You are a young Italian native speaker attending middle school.",
    # "You are brilliant Italian native speaker attending lower secondary school with high grades in Italian language",
    # "You are a clever Italian native speaker attending lower secondary school and you prefer mathematics and science to Italian language",
    # "Sei un giovane studente Italiano che frequenta le prime classi della scuola media.",
    # "Sei un brillante studente Italiano che frequenta le prime classi della scuola media con voti alti in Italiano.",
    # "Sei uno studente Italiano intelligente che frequenta le prime classi della scuola media e preferisci matematica e scienze all'italiano.",
    # "You are a famous Italian writet known for his stylish and elegant prose",
]

USER_PROMPTS_CItA = [
    "Scrivi un testo estremamente breve in cui sviluppi il seguente argomento:\nArgomento:\n{m}\n\nNello scrivere non usare argomenti eccessivamente fantasiosi o irrealistici, ma cerca di mantenere un tono serio e fai riferimento a fatti o situazioni realy. Puoi anche scrivere in prima persona.",
]

def get_random_prompt_cita(m):
    return [
        {"role":"system", "content":random.choice(SYSTEM_PROMPTS_CItA)},
        {"role":"user", "content":random.choice(USER_PROMPTS_CItA).format(m=m)}
    ]

SYSTEM_PROMPTS_ANITA = [
    # "You are a young Italian native speaker attending middle school",
    # "You are brilliant Italian native speaker attending lower secondary school with high grades in Italian language",
    # "You are a clever Italian native speaker attending lower secondary school and you prefer mathematics and science to Italian language",
    "Sei un giovane studente Italiano che frequenta le prime classi della scuola media.",
    "Sei un brillante studente Italiano che frequenta le prime classi della scuola media con voti alti in Italiano.",
    "Sei uno studente Italiano intelligente che frequenta le prime classi della scuola media e preferisci matematica e scienze all'italiano.",
    # "You are a famous Italian writet known for his stylish and elegant prose",
]

def get_random_prompt_cita_anita(m):
    return [
        {"role":"system", "content":random.choice(SYSTEM_PROMPTS_ANITA)},
        {"role":"user", "content":random.choice(USER_PROMPTS_CItA).format(m=m)}
    ]


SYSTEM_PROMPTS_XSUM=[
    "You are a journalist from the United Kingdom writing for a national newspaper on a broad range of topics",
    "You are a journalist from the Wales writing for a national newspaper on a broad range of topics, with a focus on local news",
    "You are a journalist from the United States writing for an international newspaper on a broad range of topics, Economics and Politics",
    "You are a journalist from the United Kingdom writing for a national newspaper on a broad range of topics, with a focus on Sports",
]

USER_PROMPTS_XSUM=[
    "Write a piece of news, that will appear in a national newspapers in the UK and that has the following title:\n\n{m}",
]

USER_PROMPTS_XSUM_INFORMED=[
    "Write a piece of news, that will appear in a national newspapers in the UK and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 500 words.",
]

def get_random_prompt_xsum(m, informed):
    out = [{"role":"system", "content":random.choice(SYSTEM_PROMPTS_XSUM)},]
    if not informed:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM).format(m=m)})
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM_INFORMED).format(m=m)})
    return out

def get_random_prompt_xsum_anita(m, informed):
    raise NotImplementedError("Not implemented yet")