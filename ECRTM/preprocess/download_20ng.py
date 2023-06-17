import os
from optparse import OptionParser
from sklearn.datasets import fetch_20newsgroups

import sys
sys.path.append('.')

from utils.data import file_utils


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    names = ['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism']

    prefixes = set([name.split('.')[0] for name in names])

    groups = {'20ng_all': names}
    for prefix in prefixes:
        elements = [name for name in names if name.split('.')[0] == prefix]
        groups['20ng_' + prefix] = elements

    print(groups)

    for group in groups.keys():
        if len(groups[group]) > 1:
            for subset in ['train', 'test', 'all']:
                download_articles(group, groups[group], subset)


def download_articles(name, categories, subset):

    print("===>name: ", name)
    print("===>categories: ", categories)
    print("===>subset: ", subset)

    data = []
    print("Downloading articles")
    newsgroups_data = fetch_20newsgroups(subset=subset, categories=categories, remove=())

    for i in range(len(newsgroups_data['data'])):
        line = newsgroups_data['data'][i]
        data.append({'text': line, 'group': newsgroups_data['target_names'][newsgroups_data['target'][i]]})

    print(len(data))
    raw_data_dir = os.path.join('../data/raw_data', '20ng', name)
    print("Saving to", raw_data_dir)
    file_utils.make_dir(raw_data_dir)
    file_utils.save_jsonlist(data, os.path.join(raw_data_dir, subset + '.jsonlist'))

if __name__ == '__main__':
    main()
