import spacy
from collections import Counter

# 日本語モデルをロード (事前にインストールが必要: `python -m spacy download ja_core_news_sm`)
nlp = spacy.blank("ja")
nlp.max_length = 20000000  # 最大長を設定


def generate_string(text, length=20):
    # 入力テキストを処理
    doc = nlp(text)
    chars = [char.text for char in doc if not char.is_space]  # 空白を除外

    # 1-gramの頻度を計算
    unigram_counts = Counter(chars)
    most_common_char = unigram_counts.most_common(1)[0][0]  # 最も頻度の高い文字を取得
    generated_string = most_common_char
    bigrams = [chars[i : i + 2] for i in range(len(chars) - 1)]
    bigram_counts = Counter(["".join(bigram) for bigram in bigrams])

    # 文字列生成ループ
    for _ in range(length - 1):
        # 2-gramの頻度を計算

        # 現在の文字から始まる2-gramをフィルタリング
        filtered_bigrams = {
            k: v for k, v in bigram_counts.items() if k.startswith(most_common_char)
        }
        if not filtered_bigrams:
            break  # フィルタリング結果が空なら終了

        # 最も頻度の高い2-gramを取得
        most_common_bigram = max(filtered_bigrams, key=filtered_bigrams.get)
        most_common_char = most_common_bigram[1]  # 2-gramの2文字目を取得
        generated_string += most_common_char

    return generated_string


# ファイルからテキストを読み込む
with open("aozora-small.txt", "r", encoding="utf-8") as f:
    sample_text = f.read()

# テキストを分割して処理
chunk_size = 4000  # SudachiPyの制限に合わせたチャンクサイズ（49149バイト以下に設定）
chunks = [
    sample_text[i : i + chunk_size] for i in range(0, len(sample_text), chunk_size)
]

# 各チャンクを処理して結果を結合
processed_text = ""
for chunk in chunks:
    processed_text += generate_string(chunk, length=20)

# 結果を出力
print("生成された文字列:", processed_text)
