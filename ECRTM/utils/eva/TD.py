import argparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    return args


def TD_eva(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD


def compute_TD(data_path):
    texts = list()
    with open(data_path, 'r') as file:
        for line in file:
            texts.append(line.strip())

    n_top_words = len(texts[0].split())
    TD = TD_eva(texts)
    print(f"===>TD_T{n_top_words}: {TD:.5f}")


if __name__ == "__main__":
    args = parse_args()
    compute_TD(args.data_path)
