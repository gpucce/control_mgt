import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import io
import os 
import stanza
import json
import zipfile
import pandas as pd

from multiprocessing import Pool
from stanza.utils.conll import CoNLL
from tqdm import tqdm
from transformers import AutoTokenizer

from ProfilingUD import ling_monitoring

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_data(path):
    if path.endswith(".json"):
        try:
            data = json.load(open(path)) 
        except:
            data = pd.read_json(path, lines=True)   # FIXME
            data = data.to_dict(orient="records")
    else:
        data = pd.read_json(path, lines=True)
        data = data.to_dict(orient="records")
    print(f"- data length: {len(data)}")
    return data


def run_parser(data, method="dpo-llama", device="cuda"):
    nlp = stanza.Pipeline("en", verbose=False, device=device, processors="tokenize,mwt,pos,lemma,depparse")

    conllu_dict = {}
    for line in tqdm(data):
        doc_id = line["doc-id"]
        text = line[method]
        doc = nlp(text)
        conllu_dict[doc_id] = CoNLL.convert_dict(doc.to_dict())
    
    return conllu_dict
    

def _io_save_coll(doc_id, conll_doc):
    output = io.StringIO()
    counter = 1
    for sentence in conll_doc:
        output.write("# sent_id = " + str(doc_id) + "_" + str(counter) + "\n")
        output.write("# text = None" + "\n")
        for token in sentence:
            output.write("\t".join(token) + "\n")
        output.write("\n")
        counter += 1
    result = output.getvalue()
    output.close()
    return result


def process_chunk(args):
    chunk, method, device = args
    conllu_archive = run_parser(chunk, method=method, device=device)
    return conllu_archive

def get_outdir(args):
    datapath = args.datapath
    basedir = os.path.join("ilc_profiler", "parsed")
    adapter_name = datapath.split("/")[-2]
    model_name = datapath.split("/")[-3]    # FIXME
    if "xsum-iter-1" in datapath:
        outdir = os.path.join(basedir, "xsum", "dpo-iter1", model_name, adapter_name)
    elif "xsum-iter-2" in datapath:
        outdir = os.path.join(basedir, "xsum", "dpo-iter2", model_name, adapter_name)
    elif "xsum-naive-iter-1" in datapath:
        outdir = os.path.join(basedir, "xsum", "dpo-iter1-naive", model_name, adapter_name)
    elif "xsum-naive-iter-2" in datapath:
        outdir = os.path.join(basedir, "xsum", "dpo-iter2-naive", model_name, adapter_name)
    elif "m4abs-iter-1" in datapath:
        outdir = os.path.join(basedir, "m4abs", "dpo-iter1", model_name, adapter_name)
    elif "m4abs-iter-2" in datapath:
        outdir = os.path.join(basedir, "m4abs", "dpo-iter2", model_name, adapter_name)
    elif "m4abs-naive-iter-1" in datapath:
        outdir = os.path.join(basedir, "m4abs", "dpo-iter1-naive", model_name, adapter_name)
    elif "m4abs-naive-iter-2" in datapath:
        outdir = os.path.join(basedir, "m4abs", "dpo-iter2-naive", model_name, adapter_name)
    # elif "xsum" in datapath:
    #     basedir = os.path.join(basedir, "xsum", "vanilla", model_name)  # FIXME model_name is correct? (gemma OR llama)
    return outdir

def main(args):
    data = get_data(args.datapath)
    parser_outdir = get_outdir(args)
    
    if args.max_length is not None:
        parser_outdir += f"-cut{args.max_length}"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=os.getenv("MY_HF_TOKEN"))
        texts = [t[args.method] for t in data]
        texts = tokenizer.batch_decode(tokenizer(texts, max_length=256, truncation=True).input_ids, skip_special_tokens=True)
        for i, t in enumerate(texts):
            data[i][args.method] = t
    
    os.makedirs(parser_outdir, exist_ok=True)
    
    if args.testset_only:
        archive_fn = os.path.join(parser_outdir, "testset_parsing.conllu.zip")
        ling_monitoring_fn = "testset_" + args.method
    else:
        archive_fn = os.path.join(parser_outdir, "parsing.conllu.zip")
        ling_monitoring_fn = args.method
    
    print(f"- parser outdir: {parser_outdir}")
    print(f"- parsing archive: {archive_fn}")

    if not args.skip_parse:
        conllu_archive = run_parser(data, method=args.method, device=args.device)
        with zipfile.ZipFile(archive_fn, "w", zipfile.ZIP_DEFLATED) as zipf:
            for doc_id, data in conllu_archive.items():
                out = _io_save_coll(doc_id=doc_id, conll_doc=data)
                zipf.writestr(f"{doc_id}.conllu", out)

    ling_monitoring.run(datapath=archive_fn, output_name=ling_monitoring_fn, multisent=1, outdir=parser_outdir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/adversarial-dpo-iter1-filtered/2025-01-28-18-49/xsum-alldata-250128_223602.json")
    parser.add_argument("--skip_parse", action="store_true")
    parser.add_argument("--method", type=str, default="dpo-llama-1st-iter")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--testset_only", action="store_true")
    args = parser.parse_args()
    main(args)