__author__ = 'mangate'

import cPickle
from model import Model
import process_data
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# from client
from grpc.beta import implementations
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# 172.17.0.3
tf.app.flags.DEFINE_string('server', '172.17.0.3:9001',
                           'inception_inference service host:port')

tf.flags.DEFINE_boolean("random",False,"Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static",False,"Keep the word embeddings static (default: False)")
FLAGS =tf.flags.FLAGS

def evaluate(x, num_classes = 2, k_fold = 10):   
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
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
    test_data, test_labels = cnn_model.run(revs, embedding, word_idx_map, max_l, k_fold)
    return test_data, test_labels

def send_request(test_data):
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cnn_tweet'
    request.model_spec.signature_name = 'predict_images'
	
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_data, shape=[629, 48]))
		
    request.inputs['prob'].dtype = types_pb2.DT_FLOAT
    request.inputs['prob'].float_val.append(1.0)
	
    request.output_filter.append('scores')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    print stub.Predict(request, 5.0)  # 5 secs timeout

def evaluate_main():
    process_data.process_data("data/processed/tweet.p")
    x = cPickle.load(open("data/processed/tweet.p", "rb"))
    test_data, test_labels = evaluate(x, 2, 10)
    send_request(test_data)
    # print(test_labels)

if __name__=="__main__":
    evaluate_main()
    