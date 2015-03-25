__author__ = 'KattaAnil'

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
import matplotlib.pyplot as plt

PROTO_DICT = {'http':0, 'gnutella':1, "edonkey":2, "bittorrent":3}

def read_file_return_numpy_array(file_name):
    return np.genfromtxt(file_name, delimiter=",")

def return_train_test_labels(input_data, number_of_each_flows, percent=0.9):
    input_data = np.copy(input_data)
    train_data = None
    test_data = None
    train_labels = None
    test_labels = None
    http_data = input_data[input_data[:, 14] == PROTO_DICT['http']]
    gnutella_data = input_data[input_data[:, 14] == PROTO_DICT['gnutella']]
    edonkey_data = input_data[input_data[:, 14] == PROTO_DICT['edonkey']]
    bittorrent_data = input_data[input_data[:, 14] == PROTO_DICT['bittorrent']]

    http_labels = http_data[:, -1]
    http_data = http_data[:, :-1]
    http_labels = http_labels.reshape((http_data.shape[0], 1))

    gnutella_labels = gnutella_data[:, -1]
    gnutella_data = gnutella_data[:, :-1]
    gnutella_labels = gnutella_labels.reshape((gnutella_data.shape[0], 1))

    edonkey_labels = edonkey_data[:, -1]
    edonkey_data = edonkey_data[:, :-1]
    edonkey_labels = edonkey_labels.reshape((edonkey_data.shape[0], 1))

    bittorrent_labels = bittorrent_data[:, -1]
    bittorrent_data = bittorrent_data[:, :-1]
    bittorrent_labels = bittorrent_labels.reshape((bittorrent_data.shape[0], 1))

    if number_of_each_flows:
        http_data = http_data[:number_of_each_flows + 1, :]
        gnutella_data = gnutella_data[:number_of_each_flows + 1, :]
        edonkey_data = edonkey_data[:number_of_each_flows + 1, :]
        bittorrent_data = bittorrent_data[:number_of_each_flows + 1, :]

        http_labels = http_labels[:number_of_each_flows + 1, :]
        gnutella_labels = gnutella_labels[:number_of_each_flows + 1, :]
        edonkey_labels = edonkey_labels[:number_of_each_flows + 1, :]
        bittorrent_labels = bittorrent_labels[:number_of_each_flows + 1, :]



    print(http_data.shape)
    print(http_labels.shape)
    http_train_data = http_data[:int(http_data.shape[0]*percent), :]
    http_test_data = http_data[int(http_data.shape[0]*percent): , :]
    http_train_labels = http_labels[:int(http_labels.shape[0]*percent), :]
    http_test_labels = http_labels[int(http_labels.shape[0]*percent):, :]
    gnutella_train_data = gnutella_data[:int(gnutella_data.shape[0]*percent), :]
    gnutella_test_data = gnutella_data[int(gnutella_data.shape[0]*percent):, :]
    gnutella_train_labels = gnutella_labels[:int(gnutella_labels.shape[0]*percent), :]
    gnutella_test_labels = gnutella_labels[int(gnutella_labels.shape[0]*percent):, :]
    edonkey_train_data = edonkey_data[:int(edonkey_data.shape[0]*percent), :]
    edonkey_test_data = edonkey_data[int(edonkey_data.shape[0]*percent):, :]
    edonkey_train_labels = edonkey_labels[:int(edonkey_labels.shape[0]*percent), :]
    edonkey_test_labels = edonkey_labels[int(edonkey_labels.shape[0]*percent):, :]
    bittorrent_train_data = bittorrent_data[:int(bittorrent_data.shape[0]*percent), :]
    bittorrent_test_data = bittorrent_data[int(bittorrent_data.shape[0]*percent):, :]
    bittorrent_train_labels = bittorrent_labels[:int(bittorrent_labels.shape[0]*percent), :]
    bittorrent_test_labels = bittorrent_labels[int(bittorrent_labels.shape[0]*percent):, :]

    train_data = np.vstack((http_train_data, gnutella_train_data, edonkey_train_data, bittorrent_train_data))
    train_labels = np.vstack((http_train_labels, gnutella_train_labels, edonkey_train_labels, bittorrent_train_labels))
    test_data = np.vstack((http_test_data, gnutella_test_data, edonkey_test_data, bittorrent_test_data))
    test_labels = np.vstack((http_test_labels, gnutella_test_labels, edonkey_test_labels, bittorrent_test_labels))

    return (train_data, train_labels, test_data, test_labels)


from sklearn.naive_bayes import GaussianNB

def main():
    input_data = read_file_return_numpy_array("realinput.csv")
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None)
    results = {}
    """clf = GaussianNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("Accuracy with Naive Bayes Total labels is: "+str(accuracy_score(test_labels, pred)))"""
    for i in range(1, 14):
        X_indices = np.arange(train.shape[-1])
        feature_selection_clf = SelectKBest(chi2, k=i)
        feature_selection_clf.fit(train, train_labels)
        scores = -np.log(feature_selection_clf.pvalues_)
        #print(feature_selection_clf.pvalues_)
        #print(X_indices)
        train_new = feature_selection_clf.transform(train)
        test_new = feature_selection_clf.transform(test)
        #train = feature_selection_clf.fit_transform(train, train_labels)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
        clf.fit(train_new, train_labels)
        pred = clf.predict(test_new)
        results[i] = accuracy_score(test_labels, pred)
        print("Accuracy with Adaboost Total labels is: "+str(accuracy_score(test_labels, pred))+' with '+str(i)+' features')

        clf = GaussianNB()
        clf.fit(train_new, train_labels)
        pred = clf.predict(test_new)
        results[i] = accuracy_score(test_labels, pred)
        print("Accuracy with Naive Bayes is: "+str(accuracy_score(test_labels, pred))+' with '+str(i)+' features')


    plt.plot(results.keys(), results.values(), 'ro-')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Accuracy')
    plt.title('No.Selected features v/s Accuracy')
    plt.show()
    """train, train_labels, test, test_labels = return_train_test_labels(input_data, 100)
    clf = GaussianNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("Accuracy with 100 of each flows is: "+str(accuracy_score(test_labels, pred)))

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("Accuracy with 100 Adaboost Total labels is: "+str(accuracy_score(test_labels, pred)))"""


if __name__ == '__main__':
    main()