import os
import json

import argparse

import stanza
from stanza.utils.conll import CoNLL

def parse_arg():
    parser = argparse.ArgumentParser(description='Python code for parsing the generated sentences with stanza')
    parser.add_argument('-s', '--sentences', type=str, 
                        help='folder containing the generated sentences')

    return parser.parse_args()

def parse_sentences(sentences):
    # Loading the stanza pipeline
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    with open(sentences, "r") as f:
        for line in f:
            content = json.loads(line)
            generation = content["generated_text"]
            doc_id = content["id"]

            if os.path.exists("output_generations_8b_2_iter/" + doc_id + ".conllu"):
                continue
            else:
                doc = nlp(generation) # Applying the pipeline to the sentence

                # Convert stanza object to CoNLL-U format
                dicts = doc.to_dict()
                conll = CoNLL.convert_dict(dicts)

                output = open("output_generations_8b_2_iter/" + doc_id + ".conllu", "w")
                counter = 1
                for sentence in conll:
                    output.write("# sent_id = " + doc_id + "_" + str(counter) + "\n")
                    output.write("# text = None" + "\n")
                    for token in sentence:
                        output.write("\t".join(token) + "\n")
                    output.write("\n")
                    counter += 1
                output.close()

if __name__ == '__main__':
    args = parse_arg()
    sentences = args.sentences

    parse_sentences(sentences)