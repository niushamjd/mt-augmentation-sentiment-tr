import pandas as pd

unique_files = [
    "data/annotation/ipek_unique.tsv",
    "data/annotation/buse_unique.tsv",
    "data/annotation/niyousha_unique.tsv",
]

dfs = [pd.read_csv(f, sep="\t") for f in unique_files]
df_unique = pd.concat(dfs, ignore_index=True)

total_rows = len(df_unique)
preserved = (df_unique["sentiment_preserved"] == 1).sum()
not_preserved = (df_unique["sentiment_preserved"] == 0).sum()
pct_preserved = (preserved / total_rows) * 100

print(f"Toplam Satır: {total_rows}")
print(f"Preserved (1): {preserved} (%{pct_preserved:.1f})")
print(
    f"Not Preserved (0): {not_preserved} (%{(not_preserved / total_rows) * 100:.1f})\n"
)

print("Hata Kategorileri Dağılımı:")
failures = df_unique[df_unique["sentiment_preserved"] == 0]
failure_counts = failures["failure_category"].value_counts()

for cat, count in failure_counts.items():
    share = (count / len(failures)) * 100
    print(f"- {cat}: {count} (%{share:.1f})")