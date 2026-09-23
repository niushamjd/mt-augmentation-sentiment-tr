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

    df = df.reset_index(drop=True)
    df["id"] = df.index
    print("Final:", df.shape)
    print(df["label"].value_counts())

    n_bad = df["text"].str.contains("\t|\n|\r").sum()
    assert n_bad == 0, f"{n_bad} rows contain a tab or newline"

    os.makedirs("data/mt", exist_ok=True)

    out_tsv = "data/mt/en_reviews_20k.tsv"

    df[["id", "text", "label"]].to_csv(out_tsv, sep="\t", index=False)
    print("Wrote", out_tsv)

    out_src = "data/mt/en_reviews_20k.src"
    with open(out_src, "w", encoding="utf-8") as f:
        for text in df["text"]:
            f.write(text + "\n")
    print("Wrote", out_src)

    with open(out_src, encoding="utf-8") as f:
        n_lines = sum(1 for line in f)
    assert n_lines == len(df), f"src has {n_lines} lines but df has {len(df)} rows"
    print("src lines:", n_lines)

def prepare_turkish():
    # Source is already strictly binary (ClassLabel: negative/positive only) --
    # no neutral/3-star category exists to drop.
    ds = load_dataset("fthbrmnby/turkish_product_reviews", split="train")

    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "sentiment": "label"})
    df = df[df["text"].str.strip() != ""]
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

