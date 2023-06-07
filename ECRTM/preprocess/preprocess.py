import os
import re
import sys
import string
import argparse
import gensim.downloader

from collections import Counter

import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append('.')
from utils.data import file_utils


# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--label', default=None, help='field(s) to use as label (comma-separated): default=%default')
    parser.add_argument('--test', default=None, help='Test data (test.jsonlist): default=%default')
    parser.add_argument('--test_sample_size', default=None, type=int)
    parser.add_argument('--test_p', type=float, default=0.2,)
    parser.add_argument('--stopwords', default='snowball', help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_argument('--min-doc-count', type=int, default=0, help='Exclude words that occur in less than this number of documents')
    parser.add_argument('--max-doc-freq', type=float, default=1.0, help='Exclude words that occur in more than this proportion of documents')
    parser.add_argument('--keep-num', action="store_true", default=False, help='Keep tokens made of only numbers: default=%default')
    parser.add_argument('--keep-alphanum', action="store_true", default=False, help="Keep tokens made of a mixture of letters and numbers: default=%default")
    parser.add_argument('--strip-html', action="store_true", default=False, help='Strip HTML tags: default=%default')
    parser.add_argument('--no-lower', action="store_true", default=False, help='Do not lowercase text: default=%default')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum token length: default=%default')
    parser.add_argument('--min-term', type=int, default=1, help='Minimum term number')
    parser.add_argument('--vocab-size', type=int, default=None, help='Size of the vocabulary (by most common, following above exclusions): default=%default')
    parser.add_argument('--seed', type=int, default=42, help='Random integer seed (only relevant for choosing test set): default=%default')

    args = parser.parse_args()
    return args


def make_word_embeddings(vocab):
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

    num_found = 0
    for i, word in enumerate(tqdm(vocab, desc="===>making word embeddings")):
        try:
            key_word_list = glove_vectors.index_to_key
        except:
            key_word_list = glove_vectors.index2word

        if word in key_word_list:
            word_embeddings[i] = glove_vectors[word]
            num_found += 1

    print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(word_embeddings)


def get_stopwords(stopwords):
    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = file_utils.read_text(os.path.join('../data/stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = file_utils.read_text(os.path.join('../data/stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = file_utils.read_text(os.path.join('../data/stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}
    return stopword_set


def process_parsed_texts(parsed_texts, vocab, idx):
    texts = list()
    if idx is None:
        idx = range(len(parsed_texts))
    for i in tqdm(idx, desc="===>process parsed texts"):
        words = parsed_texts[i].split()
        text = [word for word in words if word in vocab]
        texts.append(' '.join(text))
    return texts


def main():
    args = parse_args()
    label_name = args.label

    if args.seed is not None:
        np.random.seed(args.seed)

    stopword_set = get_stopwords(args.stopwords)

    train_items = file_utils.read_jsonlist(os.path.join(args.dataset_path, 'train.jsonlist'))
    test_items = file_utils.read_jsonlist(os.path.join(args.dataset_path, 'test.jsonlist'))

    n_train = len(train_items)
    n_test = len(test_items)

    print(f"Found training documents {n_train} testing documents {n_test}")

    all_items = train_items + test_items
    n_items = len(all_items)

    if label_name is not None:
        label_set = set()
        for i, item in enumerate(all_items):
            label_set.add(str(item[label_name]))

        label_list = list(label_set)
        label_list.sort()
        n_labels = len(label_list)
        label2id = dict(zip(label_list, range(n_labels)))

        print("Found label %s with %d classes" % (label_name, n_labels))
        print("label2id: ", label2id)

    train_texts = list()
    test_texts = list()
    train_labels = list()
    test_labels = list()

    word_counts = Counter()
    doc_counts_counter = Counter()

    for i, item in enumerate(tqdm(all_items, desc="===>parse texts")):
        text = item['text']
        label = label2id[item[label_name]]

        tokens, _ = tokenize(text, strip_html=args.strip_html, lower=(not args.no_lower), keep_numbers=args.keep_num, keep_alphanum=args.keep_alphanum, min_length=args.min_length, stopwords=stopword_set)
        word_counts.update(tokens)
        doc_counts_counter.update(set(tokens))
        parsed_text = ' '.join(tokens)
        # train_texts and test_texts have been parsed.
        if i < n_train:
            train_texts.append(parsed_text)
            train_labels.append(label)
        else:
            test_texts.append(parsed_text)
            test_labels.append(label)

    words, doc_counts = zip(*doc_counts_counter.most_common())
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= args.min_doc_count and doc_freqs[i] <= args.max_doc_freq]

    # filter vocabulary
    if (args.vocab_size is not None) and (len(vocab) > args.vocab_size):
        vocab = vocab[:args.vocab_size]

    vocab.sort()

    print(f"Real vocab size: {len(vocab)}")

    print("===>convert to matrix...")
    vectorizer = CountVectorizer(vocabulary=vocab)
    bow_matrix = vectorizer.fit_transform(train_texts + test_texts)

    train_bow_matrix = bow_matrix[:len(train_texts)]
    test_bow_matrix = bow_matrix[-len(test_texts):]

    train_idx = np.where(train_bow_matrix.sum(axis=1) >= args.min_term)[0]
    test_idx = np.where(test_bow_matrix.sum(axis=1) >= args.min_term)[0]

    # randomly sample
    if args.test_sample_size:
        print("===>sample train and test sets...")

        train_num = len(train_idx)
        test_num = len(test_idx)
        test_sample_size = min(test_num, args.test_sample_size)
        train_sample_size = int((test_sample_size / args.test_p) * (1 - args.test_p))
        if train_sample_size > train_num:
            test_sample_size = int((train_num / (1 - args.test_p)) * args.test_p)
            train_sample_size = train_num

        train_idx = train_idx[np.sort(np.random.choice(train_num, train_sample_size, replace=False))]
        test_idx = test_idx[np.sort(np.random.choice(test_num, test_sample_size, replace=False))]

        print("===>sampled train size: ", len(train_idx))
        print("===>sampled test size: ", len(test_idx))

    train_bow_matrix = train_bow_matrix[train_idx]
    test_bow_matrix = test_bow_matrix[test_idx]
    train_labels = np.asarray(train_labels)[train_idx]
    test_labels = np.asarray(test_labels)[test_idx]

    train_texts = process_parsed_texts(train_texts, vocab, train_idx)
    test_texts = process_parsed_texts(test_texts, vocab, test_idx)

    word_embeddings = make_word_embeddings(vocab)

    file_utils.make_dir(args.output_dir)

    print("Real output_dir is ", args.output_dir)
    print("Real training size: ", len(train_texts))
    print("Real testing size: ", len(test_texts))
    print(f"average length of training set: {train_bow_matrix.sum(1).sum() / len(train_texts):.3f}")
    print(f"average length of testing set: {test_bow_matrix.sum(1).sum() / len(test_texts):.3f}")

    scipy.sparse.save_npz(os.path.join(args.output_dir, 'train_bow.npz'), train_bow_matrix)
    scipy.sparse.save_npz(os.path.join(args.output_dir, 'test_bow.npz'), test_bow_matrix)

    scipy.sparse.save_npz(os.path.join(args.output_dir, 'word_embeddings.npz'), word_embeddings)

    file_utils.save_text(train_texts, os.path.join(args.output_dir, 'train_texts.txt'))
    file_utils.save_text(test_texts, os.path.join(args.output_dir, 'test_texts.txt'))

    np.savetxt(os.path.join(args.output_dir, 'train_labels.txt'), train_labels, fmt='%i')
    np.savetxt(os.path.join(args.output_dir, 'test_labels.txt'), test_labels, fmt='%i')
    file_utils.save_text(vocab, os.path.join(args.output_dir, 'vocab.txt'))


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != '_']
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams

    return tokens, counts


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main()
