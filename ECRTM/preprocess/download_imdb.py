from optparse import OptionParser

import os
import errno
import tarfile

from torchvision.datasets.utils import download_url

import sys
sys.path.append('.')
from utils.data import file_utils


class IMDB:

    """`IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        strip_html (bool, optional): If True, remove html tags during preprocessing; default=True
    """
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    raw_filename = 'aclImdb_v1.tar.gz'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    unlabeled_file = 'unlabeled.jsonlist'

    def __init__(self, root, download=True):
        super().__init__()
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.unlabeled_file))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_filename))

    def download(self):
        """Download the IMDB data if it doesn't exist in processed_folder already."""

        if self._check_raw_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        download_url(self.url, root=self.root,
                     filename=self.raw_filename, md5=None)
        if not self._check_raw_exists():
            raise RuntimeError("Unable to find downloaded file. Please try again.")
        else:
            print("Download finished.")

    def preprocess(self):
        """Preprocess the raw data file"""
        if self._check_processed_exists():
            return

        train_lines = []
        test_lines = []
        unlabeled_lines = []

        print("Opening tar file")
        # read in the raw data
        tar = tarfile.open(os.path.join(self.root, self.raw_filename), "r:gz")
        # process all the data files in the archive
        print("Processing documents")
        for m_i, member in enumerate(tar.getmembers()):
            # Display occassional progress
            if (m_i + 1) % 5000 == 0:
                print("Processed {:d} / 100000".format(m_i+1))
            # get the internal file name
            parts = member.name.split(os.sep)

            if len(parts) > 3:
                split = parts[1]  # train or test
                label = parts[2]  # pos, neg, or unsup
                name = parts[3].split('.')[0]
                doc_id, rating = name.split('_')
                doc_id = int(doc_id)
                rating = int(rating)

                # read the text from the archive
                f = tar.extractfile(member)
                bytes = f.read()
                text = bytes.decode("utf-8")
                # tokenize it using spacy
                if label != 'unsup':
                    # save the text, label, and original file name
                    doc = {'id': split + '_' + str(doc_id), 'text': text, 'sentiment': label, 'orig': member.name, 'rating': rating}
                    if split == 'train':
                        train_lines.append(doc)
                    elif split == 'test':
                        test_lines.append(doc)
                else:
                    doc = {'id': 'unlabeled_' + str(doc_id), 'text': text, 'sentiment': None, 'orig': member.name, 'rating': rating}
                    unlabeled_lines.append(doc)

        print("Saving processed data to {:s}".format(self.root))
        file_utils.save_jsonlist(train_lines, os.path.join(self.root, self.train_file))
        file_utils.save_jsonlist(test_lines, os.path.join(self.root, self.test_file))
        file_utils.save_jsonlist(unlabeled_lines, os.path.join(self.root, self.unlabeled_file))


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', type=str, default='../data/raw_data/imdb',
                      help='Destination directory: default=%default')

    (options, args) = parser.parse_args()

    root_dir = options.root_dir
    IMDB(root_dir, download=True)


if __name__ == '__main__':
    main()
