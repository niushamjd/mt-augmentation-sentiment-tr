"""Shared Turkish text normalisation, used for ALL Turkish text in the project
(real reviews, synthetic reviews, and the text the classifier tokeniser is
trained on), so that real and synthetic data always look the same.

Why this exists:
- The real reviews (fthbrmnby/turkish_product_reviews) are 100% lowercase.
  Cased synthetic Turkish would give <unk> for every capital letter in our
  classifier tokeniser, and the models could learn "capitals = synthetic".
- Python's str.lower() is wrong for Turkish: "I" becomes "i" (should be "ı"),
  and "İ" becomes "i" + U+0307 (an invisible combining dot). The real data
  was lowercased this way at the source, so it contains 577 of these dots in
  real_train alone (e.g. "bi̇r", "deği̇l" from reviews written in capitals).

normalise_tr() fixes both: Turkish-aware lowercasing plus removing the stray
combining dot after i. It is idempotent: normalise_tr(normalise_tr(x)) == normalise_tr(x).
"""

COMBINING_DOT = "\u0307"


def normalise_tr(text):
    text = str(text)
    # Turkish capitals first, before .lower() can get them wrong
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    # "i" + combining dot -> plain "i" (left over from wrong lowercasing at the source)
    text = text.replace("i" + COMBINING_DOT, "i")
    return text


if __name__ == "__main__":
    # small self-test: python src/text_utils.py
    tests = {
        "İstanbul'a geldi, IŞIK yandı.": "istanbul'a geldi, ışık yandı.",
        "Bu Çok Güzel Bir Ürün": "bu çok güzel bir ürün",
        "bi\u0307r deği\u0307l": "bir değil",
        "zaten küçük harf": "zaten küçük harf",
    }
    for raw, expected in tests.items():
        out = normalise_tr(raw)
        assert out == expected, (raw, out, expected)
        assert normalise_tr(out) == out  # idempotent
        print(f"OK  {raw!r} -> {out!r}")
    print("all tests passed")
