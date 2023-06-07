import argparse
import numpy as np
import scipy.io

from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, pred):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    for func in metrics_func:
        print("===>{}: {:.5f}".format(func['name'], func['method'](labels, pred)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--label_path')
    args = parser.parse_args()

    test_theta = scipy.io.loadmat(args.path)['test_theta']
    pred = np.argmax(test_theta, axis=1)

    test_labels = np.loadtxt(args.label_path + '/test_labels.txt')

    clustering_metric(test_labels, pred)
