import MeCab
import unidic_lite
from collections import defaultdict

# MeCab 初期化
mecab = MeCab.Tagger()
mecab.parse("")  # バグ回避用

#text = "わたしははしをわたる"
#text = "くるまでまつ"
text = "そこで話は終わりになった"

def get_pos_candidates_for_substring(substring, N=50):
    """部分文字列に対して品詞候補を調査"""
    raw_result = mecab.parseNBest(N, substring)
    pos_candidates = set()
    #print(raw_result)
    for parse_block in raw_result.strip().split("EOS\n"):
        if not parse_block.strip():
            continue
        for line in parse_block.strip().split("\n"):
            if not line or "\t" not in line:
                continue
            surface, feature_str = line.split("\t", 1)
            if surface != substring:
                continue  # 完全一致する部分だけを対象にする
            features = feature_str.split()
            #base = features[2] if len(features) > 2 else ""
            pos = features[3] if len(features) > 3 else ""
            if "-" in pos:
                pos=pos.split("-")[0]
            #pos_candidates.add((pos, base))
            pos_candidates.add((pos, ""))
    return pos_candidates

def analyze_sentence(sentence, N=10):
    """文章中のすべての部分文字列に対して品詞候補を調査"""
    result=defaultdict(list)
    length = len(sentence)
    for start in range(length):
        for end in range(start + 1, length + 1):
            substring = sentence[start:end]
            candidates = get_pos_candidates_for_substring(substring, N)
            if candidates:
                print(f"[{start}:{end}] '{substring}':")
                for pos, base in sorted(candidates):
                    #result[substring].append((pos, base))
                    result[substring].append((pos, ""))
                    print(f"    - {pos}, {base}")
                #print()
    return result
# 実行例
print(f"対象文「{text}」\n")
result=analyze_sentence(text, N=10)

print(result)


def display_connections_from_text(text, result):
    length = len(text)
    visited_nodes = {}  # {pos: ノード名}

    def dfs(pos, prefix="", is_last=True, visited_path=None):
        if visited_path is None:
            visited_path = set()

        if pos >= length:
            return  # 終端

        indent = prefix + ("└─" if is_last else "├─")
        for end in range(pos + 1, length + 1):
            substring = text[pos:end]
            if substring in result:
                node_key = (pos, substring)

                # 合流ポイントチェック
                if node_key in visited_nodes:
                    marker = "↑"  # 合流記号
                    print(f"{indent}{substring}[{pos}-> ] {marker} ")
                    break
                else:
                    visited_nodes[node_key] = True

                # 品詞と語形をまとめる
                entries = result[substring]
                unique_entries = defaultdict(set)
                for pos_tag, surface in entries:
                    unique_entries[pos_tag].add(surface)

                details = ", ".join(f"{tag}: {', '.join(sorted(forms))}" for tag, forms in unique_entries.items())
                print(f"{indent}{substring}[{pos}->{end}] ({details}) ")
                prefix+=("    "*(end-pos-1))

                new_prefix = prefix + ("    " if is_last else "│   ")
                dfs(end, new_prefix, is_last=True, visited_path=visited_path | {node_key})

    dfs(0)


# 実行
print(f"【解析対象の読み】『{text}』\n")
display_connections_from_text(text, result)

connection_matrix = {
    "名詞": {"名詞": 0.6, "動詞": 0.7, "形容詞": 0.0, "助詞": 0.9, "助動詞": 0.0, "終助詞": 0.0, "記号": 0.0,
             "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.5, "文末": 0.0},
    "動詞": {"名詞": 0.8, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.9, "助動詞": 0.7, "終助詞": 0.5, "記号": 0.0,
             "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.3, "文末": 0.8},
    "形容詞": {"名詞": 0.8, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.9, "助動詞": 0.6, "終助詞": 0.5, "記号": 0.0,
               "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.4, "文末": 0.8},
    "助詞": {"名詞": 0.9, "動詞": 0.8, "形容詞": 0.8, "助詞": 0.6, "助動詞": 0.0, "終助詞": 0.0, "記号": 0.0,
             "副詞": 0.7, "接頭辞": 0.0, "接続詞": 0.5, "感動詞": 0.0, "接尾辞": 0.0, "文末": 0.0},
    "助動詞": {"名詞": 0.0, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.8, "助動詞": 0.0, "終助詞": 0.5,
               "記号": 0.0, "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0,
               "文末": 0.8},
    "終助詞": {"名詞": 0.0, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.0, "助動詞": 0.0, "終助詞": 0.0,
               "記号": 0.0, "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0,
               "文末": 0.9},
    "記号": {"名詞": 0.0, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.0, "助動詞": 0.0, "終助詞": 0.0, "記号": 0.0,
             "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0, "文末": 0.0},
    "副詞": {"名詞": 0.8, "動詞": 0.8, "形容詞": 0.8, "助詞": 0.7, "助動詞": 0.0, "終助詞": 0.0, "記号": 0.0,
             "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0, "文末": 0.0},
    "接頭辞": {"名詞": 0.9, "動詞": 0.5, "形容詞": 0.6, "助詞": 0.0, "助動詞": 0.0, "終助詞": 0.0,
               "記号": 0.0, "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0,
               "文末": 0.0},
    "接続詞": {"名詞": 0.7, "動詞": 0.7, "形容詞": 0.7, "助詞": 0.6, "助動詞": 0.0, "終助詞": 0.0, "記号": 0.0,
               "副詞": 0.7, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0, "文末": 0.0},
    "感動詞": {"名詞": 0.0, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.0, "助動詞": 0.0, "終助詞": 0.0,
               "記号": 0.0, "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0,
               "文末": 0.9},
    "接尾辞": {"名詞": 0.7, "動詞": 0.0, "形容詞": 0.0, "助詞": 0.6, "助動詞": 0.0, "終助詞": 0.0,
               "記号": 0.0, "副詞": 0.0, "接頭辞": 0.0, "接続詞": 0.0, "感動詞": 0.0, "接尾辞": 0.0,
               "文末": 0.0}
}


# 使用例

def enumerate_valid_sequences_complete(text, result, connection_matrix):
    n = len(text)
    sequences = []

    def dfs2(index, current_sequence):
        if index == n:
            sequences.append(current_sequence)
            return
        for length in range(1, n - index + 1):
            substring = text[index:index+length]
            if substring in result:
                for pos, word in result[substring]:
                    # 接続行列に存在しない場合はFalse扱い
                    if not current_sequence or connection_matrix.get(current_sequence[-1][1], {}).get(pos, False):
                        #dfs(index + length, current_sequence + [(substring,pos, word)])
                        dfs2(index + length, current_sequence + [(substring,pos)])

    dfs2(0, [])
    return sequences


output_sequences = enumerate_valid_sequences_complete(text, result, connection_matrix)
for i in output_sequences:
    print(i)


import numpy as np
import heapq

def dp_enumerate_sequences_top_k(text, result, connection_matrix, k=10):
    n = len(text)
    dp = [None] * (n + 1)  # 各位置までの最良スコアと経路リストを記録
    dp[0] = [(0.0, [])]    # スコア0、空の経路

    for i in range(n):
        if dp[i] is None:
            continue

        for current_score, current_sequence in dp[i]:
            for length in range(1, n - i + 1):
                substring = text[i:i+length]
                if substring in result:
                    for pos, word in result[substring]:
                        if not current_sequence:
                            transition_score = 1.0  # 文頭は制約なし
                        else:
                            prev_pos = current_sequence[-1][1]
                            transition_score = connection_matrix.get(prev_pos, {}).get(pos, 0.0)

                        if transition_score > 0.0:
                            new_score = current_score + np.log(transition_score + 1e-9)  # logスコア
                            next_index = i + length

                            if dp[next_index] is None:
                                dp[next_index] = []
                            heapq.heappush(dp[next_index], (new_score, current_sequence + [(substring, pos)]))

                            # 上位k個に制限
                            if len(dp[next_index]) > k:
                                heapq.heappop(dp[next_index])

    # 最終位置に到達していれば、スコアが良い順にソートして返す
    if dp[n]:
        return sorted(dp[n], reverse=True)[:k]
    else:
        return []

# 使用例
top_sequences = dp_enumerate_sequences_top_k(text, result, connection_matrix, k=100)

# 上位10件を表示
for rank, (score, seq) in enumerate(top_sequences[:10], 1):
    print(f"Top {rank}: Score = {score:.4f}, Sequence = {seq}")