from datasets import load_dataset
import pandas as pd
import argparse

def prepare_english():
    ds = load_dataset("fancyzhx/amazon_polarity", split="train[:200000]")

    df = ds.to_pandas()
    df = df.rename(columns={"content": "text"})
    print(df["text"].duplicated().sum())
    df = df.drop_duplicates(subset="text")

    print(df.shape)
    print(df["label"].value_counts())

def prepare_turkish():
    ds = load_dataset("fthbrmnby/turkish_product_reviews", split="train")

    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "sentiment": "label"})
    df = df.drop_duplicates(subset="text")

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

