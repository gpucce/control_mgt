import pandas as pd
import json

from argparse import ArgumentParser

def main(args):
    assert "alldata" in args.datapath, "alldata string not in datapath"
    data = pd.read_json(args.datapath)
    test_ids = json.load(open(args.split_path))["te"]
    data = data[data["doc-id"].isin(test_ids)]
    print(data.shape)
    out_fn = args.datapath.replace("alldata", "testset")
    print(f"- storing test set results in: {out_fn}")
    data_dict = data.to_dict(orient="records")
    with open(out_fn, "w") as jf:
        json.dump(data_dict, jf, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/andrea-dpo-iter1-filtered/2025-01-28-18-49/xsum-alldata-250128_223602.json")
    parser.add_argument("--split_path", type=str, default="data/splits/split.100000.json")
    args = parser.parse_args()
    main(args)