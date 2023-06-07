import os
import numpy as np
import scipy.io
import yaml
import argparse
import importlib
from runners.Runner import Runner

from utils.data.TextData import TextData
from utils.data import file_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-m', '--model')
    parser.add_argument('-c', '--config')
    parser.add_argument('-k', '--num_topic', type=int, default=50)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=1)
    args = parser.parse_args()
    return args


def print_topic_words(beta, vocab, num_top_word):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        print('Topic {}: {}'.format(i + 1, topic_str))
    return topic_str_list


def main():
    args = parse_args()

    # loading model configuration
    file_utils.update_args(args, path=args.config)

    output_prefix = f'output/{args.dataset}/{args.model}_K{args.num_topic}_{args.test_index}th'
    file_utils.make_dir(os.path.dirname(output_prefix))

    seperate_line_log = '=' * 70
    print(seperate_line_log)
    print(seperate_line_log)
    print('\n' + yaml.dump(vars(args), default_flow_style=False))

    dataset_handler = TextData(args.dataset, args.batch_size)

    args.vocab_size = dataset_handler.train_data.shape[1]
    args.word_embeddings = dataset_handler.word_embeddings

    # train model via runner.
    runner = Runner(args)
    beta = runner.train(dataset_handler.train_loader)

    # print and save topic words.
    topic_str_list = print_topic_words(beta, dataset_handler.vocab, num_top_word=args.num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}_T{args.num_top_word}')

    # save inferred topic distributions of training set and testing set.
    train_theta = runner.test(dataset_handler.train_data)
    test_theta = runner.test(dataset_handler.test_data)

    params_dict = {
        'beta': beta,
        'train_theta': train_theta,
        'test_theta': test_theta,
    }

    scipy.io.savemat(f'{output_prefix}_params.mat', params_dict)


if __name__ == '__main__':
    main()
