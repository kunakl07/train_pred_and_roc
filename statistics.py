import logging
import argparse
import numpy as np
import random
import sklearn
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import tensorflow as tf
import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_roc_curve(test_path, model_path, predictions, true_classes, randomlist):
    # roc curve and auc

    ns_probs = randomlist
    lr_probs = predictions[:, 0]
    # calculate scores
    ns_auc = roc_auc_score(true_classes, ns_probs)
    lr_auc = roc_auc_score(true_classes, lr_probs)
    # summarize scores
    print('Random Classifier: ROC AUC=%.3f' % (ns_auc))
    print('CNN Classifier: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(true_classes, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(true_classes, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random_Classifier')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CNN_Classifier')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def main(args):
    model_path = args.modelpath
    test_path = args.testpath
    img_width, img_height = 288, 432

    test_datagen = ImageDataGenerator(rescale=1. / 55)
    test_data_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=False)
    test_steps_per_epoch = (test_data_generator.samples / test_data_generator.batch_size)
    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())
    # generate  class dataset

    randomlist = []
    for j in range(201):
        randomlist.append(random.randint(0, 1))

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict_proba(test_data_generator)

  
    plot_roc_curve(model_path, test_path, predictions, true_classes, randomlist)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Predict which images are orcas")
    parser.add_argument(
        '-m',
        '--modelpath',
        type=str,
        help='path to saved model weights',
        required=True)
    parser.add_argument(
        '-c',
        "--testpath",
        type=str,
        help='directory with Test images',
        required=True)

    args = parser.parse_args()

    main(args)