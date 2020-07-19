import tensorflow as tf
import numpy as np
from mnist_model import Model
import os
import matplotlib.pyplot as plt
from ops import plot_confusion_matrix
from PIL import Image
import datetime
import argparse
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(suppress=True)

##CONFIG
parser = argparse.ArgumentParser()
# Optimization Parameters
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='the size of one batch')
parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='drop out rate of the last layer')

parser.add_argument('--lr_init', type=float,  default=0.001, help='the initial learning rate')
parser.add_argument('--lr_min', type=float, default=0.00001, help='the minimun learning rate')
parser.add_argument('--lr_n_decrease', type=int, default=10, help='how many times to decrease before lr should reach minimum')

# Network Parameters
parser.add_argument('--img_height', type=int, default=28, help='image height')
parser.add_argument('--img_width', type=int, default=28, help='image width')
parser.add_argument('--save_step', type=int, default=5, help='save every ... epochs')
parser.add_argument('--display_step', type=int, default=50, help='display/validate every ... steps')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes')

# Model Parameters
parser.add_argument('--load-dir', type=str, default='', help='directory to load the model')
parser.add_argument('--model', type=str, default='vgg', help='model of CNN')
parser.add_argument('--data-dir', type=str, default='MNIST_data/', help='path for MNIST data')
parser.add_argument('--evaluation', action='store_true', default=False, help='lauch evaluation without train')
args = parser.parse_args()

# General
timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
model_dir = './experiments/' + args.model + '-model-' + timestamp + '/'


def load_mnist():
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    X_train = [np.expand_dims(img.reshape([args.img_height, args.img_width]), axis=-1) for img in mnist.train.images]
    y_train = mnist.train.labels

    X_val = [np.expand_dims(img.reshape([args.img_height, args.img_width]), axis=-1) for img in mnist.validation.images]
    y_val = mnist.validation.labels

    X_test = [np.expand_dims(img.reshape([args.img_height, args.img_width]), axis=-1) for img in mnist.test.images]
    y_test = mnist.test.labels

    del mnist

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batches(X, y, batch_size):
    batch_x = []
    batch_y = []
    idx = 0
    for i, x in enumerate(X):
        batch_x.append(x)
        batch_y.append(y[i])
        idx += 1
        if idx == batch_size:
            yield batch_x, batch_y
            batch_x = []
            batch_y = []
            idx = 0
    if idx > 0:
        yield batch_x, batch_y


def panning(x, v=0, h=0):
    height = x.shape[0]
    width = x.shape[1]
    if h < 0:
        x[:, :width + h] = x[:, -h:]
        x[:, width + h:] = 0
    else:
        x[:, h:] = x[:, :width - h]
        x[:, :h] = 0
    if v < 0:
        x[:height + v, :] = x[-v:, :]
        x[height + v:, :] = 0
    else:
        x[v:, :] = x[:height - v, :]
        x[:v, :] = 0
    return x


def cutting(x, v=0, h=0):
    height = x.shape[0]
    width = x.shape[1]
    if h < 0:
        x = x[:, -h:]
    else:
        x = x[:, :width - h]
    if v < 0:
        x = x[-v:, :]
    else:
        x = x[:height - v, :]
    return x


def shuffle(X, y):
    zipped_xy = list(zip(X, y))
    np.random.shuffle(zipped_xy)
    return zip(*zipped_xy)


def train(sess, model, X, y, X_val, y_val):
    # Perform training
    print("Start training..")
    saver = tf.train.Saver()

    num_batches = int(len(X) / args.batch_size)
    if (len(X) % args.batch_size != 0):
        num_batches += 1
    step = 0
    for epoch in range(args.num_epochs):
        X, y = shuffle(X, y)
        for batch_x, batch_y in get_batches(X, y, args.batch_size):
            # Run model on batch
            loss, accuracy = model.train(sess, batch_x, batch_y, args.dropout_keep_prob, step)

            if step % args.display_step == 0 or step == 0:
                val_loss, val_accuracy = model.validate(sess, X_val, y_val, step)
                print("Epoch " + str(epoch) + ", Batch " + str(step) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(accuracy) + ", Validation Loss= " +
                      "{:.5f}".format(val_loss) + ", Validation Accuracy= " +
                      "{:.5f}".format(val_accuracy))
            step += 1
        if epoch % args.save_step == 0 and epoch != 0:
            saver.save(sess, model_dir + 'model-epoch-' + str(epoch) + '.ckpt')
            print("Model saved")

    saver.save(sess, model_dir + 'model-epoch-final.ckpt')
    print("Training done, final model saved")


def test(sess, model, X, y):
    test_loss, test_accuracy, test_confusion_matrix = model.test(sess, X, y)
    print("Test loss: ", test_loss)
    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)

    if not args.evaluation:
        # dump test results to model folder in training step
        with open(model_dir + 'evaluation.txt', "w") as file:
            print("Test loss: ", test_loss, file=file)
            print("Test Accuracy: ", test_accuracy, file=file)
            print("Test Confusion Matrix:", file=file)
            print(test_confusion_matrix, file=file)

        plot_confusion_matrix(test_confusion_matrix, classes=np.arange(0, 9))
        plt.show()


def print_parameters():
    # output number of parameters for visualizing model complexity
    total_parameters = 0
    print("Number of parameters by variable:")
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = np.prod(shape.as_list())
        print(variable.name + " " + str(shape) + ": " + str(variable_parameters))
        total_parameters += variable_parameters
    print("Total number of model parameters: " + str(total_parameters))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    # determine learning rate schedule
    num_batches = int(len(X_train) / args.batch_size)
    if (len(X_train) % args.batch_size != 0):
        num_batches += 1
    lr_decay = np.power((args.lr_min / args.lr_init), (1 / args.lr_n_decrease))
    lr_step = int(args.num_epochs * num_batches / args.lr_n_decrease)

    # create model
    sess = tf.Session()
    model = Model(sess, args.n_classes, args.img_height, args.img_width, args.lr_init,
                  lr_decay, lr_step, args.lr_min, model_dir, args.model)

    # Initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if args.load_dir != '':
        # load model
        restore_model = tf.train.Saver()
        try:
            restore_model.restore(sess, os.path.join(args.load_dir, "model-epoch-final.ckpt"))
            print("Model restored.")
        except Exception as e:
            print("Model not restored: ", str(e))
            exit(0)

    if args.evaluation:
        for i, x in enumerate(X_test):
            x = panning(x, -6, -6)
            im = Image.fromarray(np.squeeze((x*255).astype(np.uint8)), mode='L')
            im.save('./MNIST_data/test/{}_{}.png'.format(i, np.argmax(y_test[i])), format='png')
        # evaluate model
        test(sess, model, X_test, y_test)

    else:
        # train model
        train(sess, model, X_train, y_train, X_val, y_val)
        # evaluate model
        test(sess, model, X_test, y_test)

if __name__ == '__main__':
    main()