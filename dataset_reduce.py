import json
import numpy as np
from collections import Counter


if __name__ == '__main__':
    with open("./data/raw/raw_news.json", "r") as f:
        article = json.load(f)

    topics = list(map(lambda x: x["topic"], article))


    def get_depth(root):
        if not root:
            return 0
        else:
            return max(get_depth(root[k]) for k in root) + 1


    depths = list(map(get_depth, topics))

    depths = np.array(depths)
    depths = Counter(depths)
    with open("./depth.json", "w") as f:
        json.dump(depths, f)

    from tqdm import tqdm

    data = [[], [], []]
    for a in tqdm(article):
        data[get_depth(a["topic"]) - 1].append(a)

    for i in range(3):
        with open(f"./data/raw/raw_news_dep{i+1}.json", "w") as f:
            json.dump(data[i], f, ensure_ascii=False)
