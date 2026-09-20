from datasets import load_dataset
import pandas as pd
import argparse

def prepare_english():
    ds = load_dataset("fancyzhx/amazon_polarity", split="train[:200000]")
    print(ds)
    print(ds[0])

    df = ds.to_pandas()
    print(df.shape)
    print(df["label"].value_counts())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["en", "tr"])
    args = parser.parse_args()

    if args.source == "en":
        prepare_english()
    else:
        prepare_turkish()

if __name__ == "__main__":
    main()
