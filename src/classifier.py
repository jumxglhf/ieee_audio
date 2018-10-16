import glob
import os

from audioFeatureExtraction import dirWavFeatureExtraction as fe

import pandas as pd
import argparse
import math
import numpy as np
import datetime
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import GradientBoostingClassifier as GBC
from frameworks.CPLELearning import CPLELearningModel



name = 'result/result_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.csv'


# Read from csv file and pre process the data
def data_preprocessing(file_path):
    index = 0
    features = []
    label = []

    for path in file_path:
        a, b, c = fe(path, 1, 1, 0.05, 0.05, compute_beat=False)
        for example in a:
            features.append(example.tolist())
            label.append(index)
        index += 1
        print(index, " FOLDER FEATURE EXTRACTED")
    return np.asarray(features), np.asarray(label)


def data_not_label(file_path):
    features = []
    a, b, c = fe(file_path, 1, 1, 0.05, 0.05, compute_beat=False)
    for example in a:
        features.append(example.tolist())
    return np.asarray(features)


def standardize_feature(feature):
    total_mean = 0.0
    mi = min(feature)
    for example in feature:
        total_mean += float(example)
    total_mean = total_mean / float(len(feature))
    variance = 0.0
    for example in feature:
        variance += (float(example) - total_mean) * (float(example) - total_mean)
    variance = math.sqrt(variance / float(len(feature)))
    result = []
    for example in feature:
        result.append((example - mi) / variance)
    return result


# create decision tree architecture
def decision_tree(train_examples, train_labels, test_examples, test_labels, verbose):
    model = DecisionTreeClassifier(criterion="entropy", min_samples_split=10,
                                   )
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    print("CONVERGENCE: ", model.score(train_examples, train_labels))
    return score


# craete a SVC architecture
def svc(train_examples, train_labels, test_examples, test_labels, verbose):
    model = SVC(C=1.5, kernel="rbf",
                gamma=0.09)
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    print("CONVERGENCE: ", model.score(train_examples, train_labels))
    return score


# create random forest neural network architecture
def random_forest(train_examples, train_labels, test_examples, test_labels, verbose):
    # class_weight = {0: 1, 1: 2} for fail, and contrary for no_fail
    model = RandomForestClassifier(n_estimators=70, criterion="entropy",
                                   warm_start=False,
                                   min_samples_split=3, class_weight={0: 3, 1: 1}
                                   )
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    print("CONVERGENCE: ", model.score(train_examples, train_labels))
    return score


# load a model and predict
def my_predict(model, test_set):
    types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(test_set, files)))

    wav_file_list = sorted(wav_file_list)
    with open(model, 'rb') as file:
        loaded_model = pickle.load(file)
    features = data_not_label(test_set)
    labels = loaded_model.predict(features)
    count = [0,0,0,0]
    index = 0
    for label in labels:
        count[label] += 1
        if label == 0:
            print(os.path.basename(wav_file_list[index]))
        index += 1
    for number in range(0,4):
        print("THE NUMBER OF ",number, " CLASS is ", count[number])


def my_compare(test_set):
    features = data_not_label(test_set)
    types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(test_set, files)))
    wav_file_list = sorted(wav_file_list)
    with open("/Users/mingxuanju/Desktop/ieee_audio/src/svm.pkl", 'rb') as file:
        svmm = pickle.load(file)
    with open("/Users/mingxuanju/Desktop/ieee_audio/src/transductive.pkl", 'rb') as file:
        transm = pickle.load(file)
    with open("/Users/mingxuanju/Desktop/ieee_audio/src/tsvm.pkl", 'rb') as file:
        rfm = pickle.load(file)
    labels1 = svmm.predict(features)
    labels2 = transm.predict(features)
    labels3 = rfm.predict(features)
    for num in range(0,len(labels1)):
        if labels1[num] != labels2[num]:
            if labels3[num] == 0:
                print(os.path.basename(wav_file_list[num]))
        else:
            if labels1[num] == 0 :
                print(os.path.basename(wav_file_list[num]))

# create gnb neural network architecture
def gnb(train_examples, train_labels, test_examples, test_labels, verbose):
    model = GaussianNB()
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    print("CONVERGENCE: ", model.score(train_examples, train_labels))
    return score


# create mlp neural network architecture
def mlp(train_examples, train_labels, test_examples, test_labels, verbose):
    nn = MLPClassifier(activation='tanh', learning_rate='adaptive'
                       , verbose=verbose, early_stopping=False,
                       max_iter=1000, hidden_layer_sizes=(30),
                       epsilon=0.005
                       )
    nn.fit(train_examples, train_labels)
    score = nn.score(test_examples, test_labels)
    print("CONVERGENCE: ", nn.score(train_examples, train_labels))
    return score


# create svm architecture
def svm(train_examples, train_labels, test_examples, test_labels, verbose):
    model = LinearSVC(verbose=verbose, penalty='l2',
                      dual=False,
                      max_iter=3000)
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    print("CONVERGENCE: ", model.score(train_examples, train_labels))
    return score


# craete brbm neural network architecture
def brbm(train_examples, train_labels, test_examples, test_labels, verbose):
    model = BernoulliRBM(verbose=verbose)
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    return score


def knn(train_examples, train_labels, test_examples, test_labels, verbose):
    highest = 0
    param = [5, "uniform", 30, 2]
    for n_neighbors in range(5, 35, 5):
        for weight in ["uniform", "distance"]:
            for leaf_size in range(25, 35):
                for p in range(1, 4):
                    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight,
                                                 leaf_size=leaf_size, p=p, n_jobs=-1)
                    model.fit(train_examples, train_labels)
                    score = model.score(test_examples, test_labels)
                    if score > highest:
                        highest = score
                        param = [n_neighbors, weight, leaf_size, p]
    print ("BEST PARAMS: ", param)
    model = KNeighborsClassifier(n_neighbors=param[0], weights=param[1],
                                 leaf_size=param[2], p=param[3], n_jobs=-1)
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    return score


def gradient_boost(train_examples, train_labels, test_examples, test_labels, verbose):
    model = GBC(learning_rate=0.1, n_estimators=500, subsample=1.0,
                min_samples_split=2, max_depth=3)
    model.fit(train_examples, train_labels)
    score = model.score(test_examples, test_labels)
    return score


def run1(file_path):
    index = -1
    features = []
    label = []

    for path in file_path:
        a, b, c = fe(path, 1, 1, 0.05, 0.05, compute_beat=False)
        for example in a:
            features.append(example.tolist())
            label.append(index)
        index += 1
        print(index, " FOLDER FEATURE EXTRACTED")
    features = np.asarray(features)
    label = np.asarray(label)
    model = LabelPropagation(max_iter=100000)
    model.fit(features, label)
    pkl_filename = "transductive.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
        print("MODEL SAVED")


def run2(file_path):
    index = -1
    features = []
    label = []
    model = CPLELearningModel(SVC(kernel="rbf", probability=True), predict_from_probabilities=True)
    for path in file_path:
        a, b, c = fe(path, 1, 1, 0.05, 0.05, compute_beat=False)
        for example in a:
            features.append(example.tolist())
            label.append(index)
        index += 1
        print(index, " FOLDER FEATURE EXTRACTED")
    features = np.asarray(features)
    label = np.asarray(label)
    model.fit(features, label)
    pkl_filename = "tsvm.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)



def run(file_path, verbose, algorithm):
    features, label = data_preprocessing(file_path)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(features, label)
    folds = 0
    total = 0
    for train_index, test_index in skf.split(features, label):
        print("PROCESSING FOLD", folds, "OUT OF 10")
        folds += 1
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = label[train_index], label[test_index]

        if algorithm == 'MLP':
            score = mlp(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'GBC':
            score = gradient_boost(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'SVM':
            score = svm(features_train, labels_train, features_test, labels_test, verbose)
            # THIS ONE SEEMS LIKE NOT WORKING#
        elif algorithm == 'BRBM':
            score = brbm(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'GNB':
            score = gnb(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'KNN':
            score = knn(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'RF':
            score = random_forest(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'DTREE':
            score = decision_tree(features_train, labels_train, features_test, labels_test, verbose)
        elif algorithm == 'SVC':
            score = svc(features_train, labels_train, features_test, labels_test, verbose)
        print("ACCURACY FOR FOLD #", folds, "IS", score)
        total += score
    print("-----------------------------------------")
    print("AVERAGE ACCURACY: ", total / 10)
    model = LinearSVC(verbose=verbose, penalty='l2',
                      dual=False,
                      max_iter=3000)
    model.fit(features, label)
    pkl_filename = "svm.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="OAB classifier")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks",
        dest="task", metavar="")

    run = tasks.add_parser("run", help="Train the classifier, and output results")
    run1 = tasks.add_parser("transductive", help="Train the classifier, and output results")
    run1.add_argument("-i", "--input", required=True, nargs="+", help="Input csv file")
    run2 = tasks.add_parser("tsvm", help="Train the classifier, and output results")
    run2.add_argument("-i", "--input", required=True, nargs="+", help="Input csv file")
    run.add_argument("-i", "--input", required=True, nargs="+", help="Input csv file")
    run.add_argument("-v", "--verbose", type=int,
                     choices=[1, 0], required=True,
                     help="Run program silently or not")
    run.add_argument("--algorithm", required=True, choices=["MLP", "GBC","SVC", "DTREE", "KNN", "SVM", "BRBM", "GNB", "RF"],
                     help="The selected algorithm")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.task == "run":
        run(args.input, args.verbose, args.algorithm)
    if args.task == "transductive":
        run1(args.input)
    if args.task == "tsvm":
        run2(args.input)

