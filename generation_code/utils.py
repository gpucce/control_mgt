import random
from itertools import product

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

# USER_PROMPTS_XSUM=[
#     # "Write a piece of news, that will appear in a national newspaper in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a local newspaper in Ireland, focus on the declaration of withnesses to the event, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a local newspaper in Wales, focus on the declaration of withnesses to the event, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a national newspaper, focus on the declaration of the police about the event, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a national newspaper, focus on the impact this news can have on the whole country, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a local newspaper in Scotland, focus on the impact this news can have on the whole country, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a a weekly magazine in England, focus on the impact this news can have on the whole country, in the UK and that has the following title:\n\n{m}",
#     # "Write a piece of news, that will appear in a a weekly magazine in England, focus on the impact this news can have on the whole country, in the UK and that has the following title:\n\n{m}",
# ]

# USER_PROMPTS_XSUM_INFORMED=[
#     "Write a piece of news, that will appear in a national newspapers in the UK and that has the following title:\n\n{m}\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 500 words.",
# ]

XSUM_VENUES = ["local newspaper", "national newspaper", "national magazine", "international magazine", "weekly magazine", "opinion magazine", "international journal"]
COUNTRY = ["Scotland", "Wales", "Ireland", "England", "Northern Ireland", "United Kingdom", "UK"]
FOCUS = ["the whole country", "the town or city where it happened", "the world", "the local community", "the country", "Europe"]

_USER_PROMPT_XSUM="Write a piece of news, that will appear in a {venue} in {country}, focus on {focus}, and that has the following title:\n\n|||title|||"
USER_PROMPTS_XSUM=[_USER_PROMPT_XSUM.format(venue=v, country=c, focus=f).replace("|||title|||", "{m}") for v, c, f in product(XSUM_VENUES, COUNTRY, FOCUS)]

_USER_PROMPT_XSUM_INFORMED="Write a piece of news, that will appear in a {venue} in {country}, focus on {focus}, and that has the following title:\n\n|||title|||\n In writing avoid any kind of formatting, do not repeat the title and keep the text informative and not vague. You don't have to add the date of the event but you can, use at most 1000 words."
USER_PROMPTS_XSUM_INFORMED=[_USER_PROMPT_XSUM_INFORMED.format(venue=v, country=c, focus=f).replace("|||title|||", "{m}") for v, c, f in product(XSUM_VENUES, COUNTRY, FOCUS)]

def get_random_prompt_xsum(m, informed):
    out = [{"role":"system", "content":random.choice(SYSTEM_PROMPTS_XSUM)},]
    if not informed:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM).format(m=m)})
    else:
        out.append({"role":"user", "content":random.choice(USER_PROMPTS_XSUM_INFORMED).format(m=m)})
    return out

def get_random_prompt_xsum_anita(m, informed):
    raise NotImplementedError("Not implemented yet")

SYSTEM_REGESTO_PROMPTS = [
    "You are an expert philolohist who dedicated his life to the study of Latin language and has a deep expertise in the field of medieval manuscripts. You have dedicated several years of your life in learning how to write the \"regesto\" of texts in Latin."
]

USER_REGESTO_PROMPTS = [
    "Here are some examples of latin texts (TESTO ESTESO) with their corresponding regesto (REGESTO):",
    "Given the the following text in Latin\n\nTESTO ESTESO:\n{testo_esteso}.\n\nPlease write a \"regesto\" for it."
]

def get_regesto_prompt_with_example(testo_esteso, testi_esempio, regesti_esempio, n):
    assert len(testi_esempio) == len(regesti_esempio)
    rand_idxs = random.choices(range(len(testi_esempio)), k=n)
    while any([testi_esempio[i] == testo_esteso for i in rand_idxs]):
        rand_idxs = random.choices(range(len(testi_esempio)), k=n)
    testi_esempio = [testi_esempio[i] for i in rand_idxs]
    regesti_esempio = [regesti_esempio[i] for i in rand_idxs]
    return (
        USER_REGESTO_PROMPTS[0] + "\n" +
        "\n".join([f"TESTO ESTESO ({i}):\n{testi_esempio[i]}\n\nREGESTO ({i}):\n{regesti_esempio[i]}"
         for i in range(n)]) + "\n" +
        USER_REGESTO_PROMPTS[1].format(testo_esteso=testo_esteso))

def get_regesto_prompt(m, testi_estesi, regesti, n):
    out = [{"role":"system", "content":random.choice(SYSTEM_REGESTO_PROMPTS)},]
    out.append({"role":"user", "content":get_regesto_prompt_with_example(
        m, testi_estesi, regesti, n
    )})
    return out