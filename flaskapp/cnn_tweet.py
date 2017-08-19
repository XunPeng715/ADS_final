import cPickle
import tensorflow as tf
import pandas as pd

import tempfile

import warnings
warnings.filterwarnings("ignore")

# from client
from grpc.beta import implementations

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', '172.17.0.4:9001',
                           'inception_inference service host:port')
FLAGS =tf.flags.FLAGS

def getClassfication(sentence):
    x = cPickle.load(open("/flaskapp/data/processed/tweet.p", "rb"))
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    filter_h = 5
    sent_idx = get_idx_from_sent(sentence, word_idx_map, max_l, filter_h)
    result = send_request([sent_idx])
    # print result
    return result.outputs['scores'].int64_val[0]
	
'''
def getClassficationForFile(file):
    
    tempfile_path = tempfile.NamedTemporaryFile().name
    file.file.save(tempfile_path)
	
    df = pd.read_csv(tempfile_path, sep='\t')
    sent_lst = df.iloc[:,0].tolist()
    label_lst = sent_lst.map(lambda x: getClassfication(x))
    df[1] = label_lst
    df.to_csv('/flaskapp/results/result.out', index=False)
'''
	
def getTestDataOutput():
    x = cPickle.load(open("/flaskapp/data/processed/tweet.p", "rb"))
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    test_data, test_data_matrix, test_labels = make_idx_data_cv(revs, word_idx_map, 1, max_l, 5)
	
    df = pd.DataFrame()
    df['text'] = test_data
    df['label'] = test_labels

    result = send_request(test_data_matrix)
    df['predicted_label'] = result.outputs['scores'].int64_val
	
    df.to_csv('/flaskapp/results/result.out', index=False)
	
def make_idx_data_cv(revs, word_idx_map, cv, max_l, filter_h=5):
    # Transforms sentences into a 2-d matrix.

    test_data, test_data_matrix, test_labels = [], [], []

    for rev in revs:
        # convert sentence to 1 * 64 (represent by number)
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        
        if rev["split"] == cv:
            test_data.append(rev["text"])
            test_data_matrix.append(sent)
            test_labels.append(rev["y"])
    return [test_data, test_data_matrix, test_labels]
		
def get_idx_from_sent(sent, word_idx_map, max_l, filter_h=5):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x
	
def send_request(test_data):
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cnn_tweet'
    request.model_spec.signature_name = 'predict_images'
	
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_data, shape=[len(test_data), 48]))
		
    request.inputs['prob'].dtype = types_pb2.DT_FLOAT
    request.inputs['prob'].float_val.append(1.0)
	
    request.output_filter.append('scores')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub.Predict(request, 5.0)  # 5 secs timeout