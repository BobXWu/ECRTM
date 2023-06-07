import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse
import scipy.io
from utils.data import file_utils


class TextData:
    def __init__(self, dataset, batch_size):
        # train_data: NxV
        # test_data: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        dataset_path = f'../data/{dataset}'
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.vocab, self.word_embeddings = self.load_data(dataset_path)
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_data.shape[0])
        print("===>test_size: ", self.test_data.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_data.sum(1).sum() / self.train_data.shape[0]))
        print("===>#label: ", len(np.unique(self.train_labels)))

        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)
        if torch.cuda.is_available():
            self.train_data = self.train_data.cuda()
            self.test_data = self.test_data.cuda()

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self, path):

        train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        test_data = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
        test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        vocab = file_utils.read_text(f'{path}/vocab.txt')

        return train_data, test_data, train_labels, test_labels, vocab, word_embeddings
