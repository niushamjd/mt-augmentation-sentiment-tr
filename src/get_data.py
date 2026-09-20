from datasets import load_dataset
import pandas as pd
import argparse
import os

def prepare_english():
    ds = load_dataset("fancyzhx/amazon_polarity", split="train[:200000]")

    df = ds.to_pandas()
    df = df.rename(columns={"content": "text"})
    df = df.drop_duplicates(subset="text")

    print("After de-duplication:", df.shape)
    print(df["label"].value_counts())

    df["n_tokens"] = df["text"].str.split().str.len()
    print("Length distribution:")
    print(df["n_tokens"].describe())

    df = df[df["n_tokens"] <= 50]
    print("After 50-token cut:", df.shape)
    print("Class balance after cut:")
    print(df["label"].value_counts())

    neg = df[df["label"] == 0].sample(n=10000, random_state=42)
    pos = df[df["label"] == 1].sample(n=10000, random_state=42)
    df = pd.concat([neg, pos]).sample(frac=1, random_state=42)
    print("Final:", df.shape)
    print(df["label"].value_counts())

    print("Rows with tab or newline:", df["text"].str.contains("\t|\n").sum())

    out = "data/mt/en_reviews_20k.tsv"
    os.makedirs("data/mt", exist_ok=True)
    df[["text", "label"]].to_csv(out, sep="\t", index=False)
    print("Wrote", out)

def prepare_turkish():
    ds = load_dataset("fthbrmnby/turkish_product_reviews", split="train")

    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "sentiment": "label"})
    df = df.drop_duplicates(subset="text")

    print(df.shape)
    print(df["label"].value_counts())

    def take(pool, n_per_class):
        neg = pool[pool["label"] == 0].sample(n=n_per_class, random_state=42)
        pos = pool[pool["label"] == 1].sample(n=n_per_class, random_state=42)
        part = pd.concat([neg, pos]).sample(frac=1, random_state=42)
        return part, pool.drop(part.index)

    test, df = take(df, 1000)
    dev, df = take(df, 500)
    train, df = take(df, 1000)

    assert len(set(train["text"]) & set(dev["text"])) == 0
    assert len(set(train["text"]) & set(test["text"])) == 0
    assert len(set(dev["text"]) & set(test["text"])) == 0

    os.makedirs("data/sentiment", exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        path = f"data/sentiment/real_{name}.tsv"
        part[["text", "label"]].to_csv(path, sep="\t", index=False)
        print(name, part.shape, dict(part["label"].value_counts()))
    print("Leftover pool:", df.shape)


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

