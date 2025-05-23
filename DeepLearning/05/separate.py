import os
import shutil

os.makedirs("./train/cat", exist_ok=True)
os.makedirs("./train/dog", exist_ok=True)

for fname in os.listdir("./train"):
    path = os.path.join("./train", fname)
    if os.path.isfile(path):  # ファイルのみ処理
        if fname.startswith("cat"):
            shutil.move(path, f"./train/cat/{fname}")
        elif fname.startswith("dog"):
            shutil.move(path, f"./train/dog/{fname}")
