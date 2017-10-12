import gzip
import h5py
from collections import defaultdict
import numpy as np
import cPickle
from preprocess_text import clean_string
import argparse



def build_data_train(fname, vocab=None,
                     skip_headers=True, train_ratio=0.8):
    """
    Loads the data file and spits out a h5 file with record of
    {y, review_text, review_int}
    Typically two passes over the data.
    1st pass is for vocab and pre-processing. (WARNING: to get phrases, we need to go
    though multiple passes). 2nd pass is converting text into integers. We will deal with integers
    from thereafter.

    WARNING: we use h5 just as proof of concept for handling large datasets
    Datasets may fit entirely in memory as numpy as array

    """

    # create the h5 store - NOTE: hdf5 is row-oriented store and we slice rows
    # reviews_text holds the metadata and processed text file
    # reviews_int holds the ratings, ints
    h5f = h5py.File(fname + '.h5', 'w')
    shape, maxshape = (2 ** 16,), (None, )
    dt = np.dtype([('y', np.uint8),
                   ('split', np.bool),
                   ('num_words', np.uint16),
                   # WARNING: vlen=bytes in pythoon 3
                   ('text', h5py.special_dtype(vlen=str))
                   ])
    reviews_text = h5f.create_dataset('reviews', shape=shape, maxshape=maxshape,
                                      dtype=dt, compression='gzip')
    reviews_train = h5f.create_dataset(
        'train', shape=shape, maxshape=maxshape,
        dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

    reviews_valid = h5f.create_dataset(
        'valid', shape=shape, maxshape=maxshape,
        dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

    wdata = np.zeros((1, ), dtype=dt)

    # init vocab only for train data
    build_vocab = False
    if vocab is None:
        vocab = defaultdict(int)
        build_vocab = True
    nsamples = 0

    # open the file, skip the headers if needed
    f = open(fname, 'r')
    if skip_headers:
        f.readline()

    for i, line in enumerate(f):
        _, rating, review = line.strip().split('\t')

        # clean the review

        review = clean_string(review)
        review_words = review.strip().split()
        num_words = len(review_words)
        split = int(np.random.rand() < train_ratio)

        # create record
        wdata['y'] = int(float(rating))
        wdata['text'] = review
        wdata['num_words'] = num_words
        wdata['split'] = split
        reviews_text[i] = wdata

        # update the vocab if needed
        if build_vocab:
            for word in review_words:
                vocab[word] += 1

        nsamples += 1

    # histogram of class labels, sentence length
    ratings, counts = np.unique(
        reviews_text['y'][:nsamples], return_counts=True)
    sen_len, sen_len_counts = np.unique(
        reviews_text['num_words'][:nsamples], return_counts=True)
    vocab_size = len(vocab)
    nclass = len(ratings)
    reviews_text.attrs['vocab_size'] = vocab_size
    reviews_text.attrs['nrows'] = nsamples
    reviews_text.attrs['nclass'] = nclass
    reviews_text.attrs['class_distribution'] = counts
    print "vocabulary size - ", vocab_size
    print "# of samples - ", nsamples
    print "# of classes", nclass
    print "class distribution - ", ratings, counts
    sen_counts = zip(sen_len, sen_len_counts)
    sen_counts = sorted(sen_counts, key=lambda kv: kv[1], reverse=True)
    print "sentence length - ", len(sen_len), sen_len, sen_len_counts

    # WARNING: assume vocab is of order ~4-5 million words.
    # sort the vocab , re-assign ids by its frequency. Useful for downstream tasks
    # only done for train data
    if build_vocab:
        vocab_sorted = sorted(
            vocab.items(), key=lambda kv: kv[1], reverse=True)
        vocab = {}
        for i, t in enumerate(zip(*vocab_sorted)[0]):
            vocab[t] = i

    # map text to integers
    ntrain = 0
    nvalid = 0
    for i in range(nsamples):
        text = reviews_text[i]['text']
        y = int(reviews_text[i]['y'])
        split = reviews_text[i]['split']
        text_int = [y] + [vocab[t] for t in text.strip().split()]
        if split:
            reviews_train[ntrain] = text_int
            ntrain += 1
        else:
            reviews_valid[nvalid] = text_int
            nvalid += 1
    reviews_text.attrs['ntrain'] = ntrain
    reviews_text.attrs['nvalid'] = nvalid
    print "# of train - {0}, # of valid - {1}".format(reviews_text.attrs['ntrain'],
                                                      reviews_text.attrs['nvalid'])
    # close open files
    h5f.close()
    f.close()

    return vocab


def main(reviews_fname='labeledTrainData.tsv'):

    # pre-process the data
    vocab = build_data_train(reviews_fname, skip_headers=True)
    rev_vocab = {}
    for wrd, wrd_id in vocab.iteritems():
        rev_vocab[wrd_id] = wrd
    cPickle.dump((vocab, rev_vocab), open(reviews_fname + '.vocab', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fname', default='labeledTrainData.tsv',
                                         help='Name of labeled train file')
    args = parser.parse_args()
    reviews_fname = args.fname
    main(reviews_fname)

