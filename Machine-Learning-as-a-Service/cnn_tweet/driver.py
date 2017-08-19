__author__ = 'mangate'

import cPickle
from model import Model
import process_data
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

#Flags
tf.flags.DEFINE_boolean("random",False,"Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static",False,"Keep the word embeddings static (default: False)")
FLAGS =tf.flags.FLAGS

def evaluate(x, num_classes = 2, k_fold = 10):
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    # print(revs)
    # print(W2)
    if FLAGS.random:
        embedding = W2
    embedding_dim = 300
    vocab_size = len(vocab) + 1
    filter_sizes = [3, 4, 5]
    num_filters = 100
    vector_length = max_l + 2 * 4
    cnn_model = Model()
    trainable = not FLAGS.static
    cnn_model.build_model(embedding_dim, vocab_size, filter_sizes, num_filters, vector_length, num_classes, trainable)
    cnn_model.run(revs, embedding, word_idx_map, max_l, k_fold)


def evaluate_main():
    process_data.process_data("data/processed/tweet.p")
    x = cPickle.load(open("data/processed/tweet.p", "rb"))
    evaluate(x, 2, 10)

if __name__=="__main__":
    evaluate_main()
    