import pandas as pd
import tensorflow as tf
from preprocess import CleanData
from mlp_model import feed_forward, next_batch

# load data
df = pd.read_csv("data/features.csv")
X = df.iloc[:, :-2].values
Y = df.iloc[:, -1].values

# preprocess data
cleaner = CleanData(n_categories=5, n_test=0.3)
x_train, x_test, y_train, y_test = cleaner.pre_process(X, Y, dummy=True)

# define model parameters
learning_rate = 0.1
num_epochs = 1000
batch_size = 128
display_epoch = 100
n_hidden_1 = 256
n_hidden_2 = 256
num_input = x_train.shape[1]
num_classes = y_train.shape[1]

# make tensor placeholders for the input and output
X = tf.placeholder(tf.float32, shape=(None, num_input))
Y = tf.placeholder(tf.float32, shape=(None, num_classes))

# make variable tensors for the weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal(shape=(num_input, n_hidden_1)), name="h1"),
    'h2': tf.Variable(tf.random_normal(shape=(n_hidden_1, n_hidden_2)), name="h2"),
    'out': tf.Variable(tf.random_normal(shape=(n_hidden_2, num_classes)), name="wout")
}
biases = {
    'b1': tf.Variable(tf.random_normal(shape=(n_hidden_1,)), name="b1"),
    'b2': tf.Variable(tf.random_normal(shape=(n_hidden_2,)), name="b2"),
    'out': tf.Variable(tf.random_normal(shape=(num_classes,)), name="bout")
}

# Using the feed_forward operation, get the predictions
logits = feed_forward(X, weights, biases)

# Define the loss operation and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

# Define the optimizer operation
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Define a custom accuracy operation
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for epoch in range(1, num_epochs + 1):
        # get a minibatch to train on
        batch_x, batch_y = next_batch(batch_size, x_train, y_train)

        # Run the training operation to update the weights, use a feed_dict to use batch_x and batch_y
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # display output if desired
        if epoch % display_epoch == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            # You can run multiple operations using a list
            # as above, use a feed dictionary for batch_x, batch_y
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Epoch " + str(epoch) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Run the accuracy operations for the test frames
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: x_test,
                                        Y: y_test}))

    save_path = saver.save(sess, "tmp/model", global_step=500)
    print("Model saved in file: %s" % save_path)


    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph('tmp/model-500.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
    #     graph = tf.get_default_graph()
    #
    #     weights = {'h1': graph.get_tensor_by_name("h1:0"),
    #                'h2': graph.get_tensor_by_name("h2:0"),
    #                'wout': graph.get_tensor_by_name("wout:0")}
    #
    #     biases = {'b1': graph.get_tensor_by_name("b1:0"),
    #               'b2': graph.get_tensor_by_name("b2:0"),
    #               'bout': graph.get_tensor_by_name("bout:0")}
    #
    #     x = features.reshape(90, 1)
    #
    #     out = predict(tf.cast(x, tf.float32), weights, biases)
    #     indx = sess.run(tf.argmax(out, 1))
    #