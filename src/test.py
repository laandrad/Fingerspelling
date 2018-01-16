import tensorflow as tf
import pandas as pd
from mlp_model import predict
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/features.csv")
X = df.iloc[:, :-2].values
Y = df.iloc[:, -1].values
label_encode = LabelEncoder()
Y = label_encode.fit_transform(Y)
labels = ["A", "B", "C", "D", "E"]

# load model
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('tmp/model-500.meta')
    saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
    graph = tf.get_default_graph()

    weights = {'h1': graph.get_tensor_by_name("h1:0"),
               'h2': graph.get_tensor_by_name("h2:0"),
               'wout': graph.get_tensor_by_name("wout:0")}

    biases = {'b1': graph.get_tensor_by_name("b1:0"),
              'b2': graph.get_tensor_by_name("b2:0"),
              'bout': graph.get_tensor_by_name("bout:0")}

    out = predict(tf.cast(X, tf.float32), weights, biases)
    correct_pred = tf.equal(tf.argmax(out, 1), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(sess.run(accuracy))

    # indx = sess.run(tf.argmax(out, 1))
    # print(indx)

    # print("Predicted letter: {}".format(labels[int(indx[0])]))
    # print("Original letter: {}".format(df.iloc[0, -1]))

    weight_params = {'h1': sess.run(graph.get_tensor_by_name("h1:0")),
              'h2': sess.run(graph.get_tensor_by_name("h2:0")),
              'wout': sess.run(graph.get_tensor_by_name("wout:0"))}

    biases_params = {'b1': sess.run(graph.get_tensor_by_name("b1:0")),
                     'b2': sess.run(graph.get_tensor_by_name("b2:0")),
                     'bout': sess.run(graph.get_tensor_by_name("bout:0"))}

pd.to_pickle(weight_params, "tmp/weights")
pd.to_pickle(biases_params, "tmp/biases")




