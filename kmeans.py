__author__ = 'KattaAnil'


import numpy as np
from sklearn.cluster import KMeans
import csv
import itertools
import matplotlib.pyplot as plt

PROTO_DICT_MAP = {0: 'http', 1: 'gnutella', 2: "edonkey", 3: "bittorrent", 4: "skype"}

PROTO_DICT = {'http': 0.0, 'gnutella': 1.0, "edonkey": 2.0, "bittorrent": 3.0, "skype":4.0}

DEBUG = False

def get_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")


def get_attribute_group_modified(data, attribute_list):
    if len(attribute_list) == 1:
        return_data = data[:,attribute_list]
        return_data = return_data.reshape((return_data.shape[0], 1))
        return return_data
    else:
        return_data = data[:, attribute_list]
        return return_data

def return_train_test_labels(input_data, number_of_each_flows, percent=0.9):
    input_data = np.copy(input_data)
    http_data = input_data[input_data[:, 17] == PROTO_DICT['http']]
    gnutella_data = input_data[input_data[:, 17] == PROTO_DICT['gnutella']]
    edonkey_data = input_data[input_data[:, 17] == PROTO_DICT['edonkey']]
    bittorrent_data = input_data[input_data[:, 17] == PROTO_DICT['bittorrent']]
    unknown_data = input_data[input_data[:, 17] == PROTO_DICT['skype']]

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

    unknown_labels = unknown_data[:, -1]
    unknown_data = unknown_data[:, :-1]
    unknown_labels = unknown_labels.reshape((unknown_data.shape[0], 1))

    if number_of_each_flows:
        http_data = http_data[:number_of_each_flows + 1, :]
        gnutella_data = gnutella_data[:number_of_each_flows + 1, :]
        edonkey_data = edonkey_data[:number_of_each_flows + 1, :]
        bittorrent_data = bittorrent_data[:number_of_each_flows + 1, :]
        unknown_data = unknown_data[:number_of_each_flows + 1, :]

        http_labels = http_labels[:number_of_each_flows + 1, :]
        gnutella_labels = gnutella_labels[:number_of_each_flows + 1, :]
        edonkey_labels = edonkey_labels[:number_of_each_flows + 1, :]
        bittorrent_labels = bittorrent_labels[:number_of_each_flows + 1, :]
        unknown_labels = unknown_labels[:number_of_each_flows + 1, :]

    http_train_data = http_data[:int(http_data.shape[0] * percent), :]
    http_test_data = http_data[int(http_data.shape[0] * percent): , :]
    http_train_labels = http_labels[:int(http_labels.shape[0] * percent), :]
    http_test_labels = http_labels[int(http_labels.shape[0] * percent):, :]
    gnutella_train_data = gnutella_data[:int(gnutella_data.shape[0] * percent), :]
    gnutella_test_data = gnutella_data[int(gnutella_data.shape[0] * percent):, :]
    gnutella_train_labels = gnutella_labels[:int(gnutella_labels.shape[0] * percent), :]
    gnutella_test_labels = gnutella_labels[int(gnutella_labels.shape[0] * percent):, :]
    edonkey_train_data = edonkey_data[:int(edonkey_data.shape[0] * percent), :]
    edonkey_test_data = edonkey_data[int(edonkey_data.shape[0] * percent):, :]
    edonkey_train_labels = edonkey_labels[:int(edonkey_labels.shape[0] * percent), :]
    edonkey_test_labels = edonkey_labels[int(edonkey_labels.shape[0] * percent):, :]
    bittorrent_train_data = bittorrent_data[:int(bittorrent_data.shape[0] * percent), :]
    bittorrent_test_data = bittorrent_data[int(bittorrent_data.shape[0] * percent):, :]
    bittorrent_train_labels = bittorrent_labels[:int(bittorrent_labels.shape[0] * percent), :]
    bittorrent_test_labels = bittorrent_labels[int(bittorrent_labels.shape[0] * percent):, :]

    unknown_train_data = unknown_data[:int(unknown_data.shape[0] * percent), :]
    unknown_test_data = unknown_data[int(unknown_data.shape[0] * percent):, :]
    unknown_train_labels = unknown_labels[:int(unknown_labels.shape[0] * percent), :]
    unknown_test_labels = unknown_labels[int(unknown_labels.shape[0] * percent):, :]

    train_data = np.vstack((http_train_data, gnutella_train_data, edonkey_train_data, bittorrent_train_data, unknown_train_data))
    train_labels = np.vstack((http_train_labels, gnutella_train_labels, edonkey_train_labels, bittorrent_train_labels, unknown_train_labels))
    test_data = np.vstack((http_test_data, gnutella_test_data, edonkey_test_data, bittorrent_test_data, unknown_test_data))
    test_labels = np.vstack((http_test_labels, gnutella_test_labels, edonkey_test_labels, bittorrent_test_labels, unknown_test_labels))

    return (train_data, train_labels, test_data, test_labels)

class TrainFlowResult(object):
    def __init__(self):
        self.group = None
        self.probability_for_http = None
        self.probability_for_gnutella = None
        self.probability_for_edonkey = None
        self.probability_for_bittorrent = None
        self.probability_for_skype = None
        self.final_probability = None
        self.population_fraction = None
        self.application = None
        self.number_of_flows = None

    def calculate_final_probability(self):
        self.final_probability = max(self.probability_for_http, self.probability_for_gnutella, self.probability_for_edonkey, 
                                     self.probability_for_bittorrent, self.probability_for_skype)
        if self.final_probability == self.probability_for_http:
            self.application = 0.0
        elif self.final_probability == self.probability_for_gnutella:
            self.application = 1.0
        elif self.final_probability == self.probability_for_edonkey:
            self.application = 2.0
        elif self.final_probability == self.probability_for_bittorrent:
            self.application = 3.0
        elif self.final_probability == self.probability_for_skype:
            self.application = 4.0

    def get_max_prob(self):
        return max(self.probability_for_http, self.probability_for_gnutella, self.probability_for_edonkey, 
                                     self.probability_for_bittorrent, self.probability_for_skype)

    def calculate_population_fraction(self):
        self.population_fraction = max(self.probability_for_bittorrent, self.probability_for_edonkey, 
                                        self.probability_for_gnutella, self.probability_for_http, self.probability_for_skype)

def generate_stats_for_cluster(data, cluster_number):
    result_map = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in data.tolist()[0]:
        if i in result_map:
            result_map[int(i)] = result_map[i] + 1
        else:
            result_map[int(i)] = 1
    st = '{0:15}{1:1.3f}{2:10.3f}{3:10.3f}{4:10.3f}'
    st_http = None
    st_gnutella = None
    st_edonkey = None
    st_bittorrent = None
    st_skype = None
    for i in result_map:
        if i == PROTO_DICT['http'] and sum(result_map.values()) != 0:
             st_http = float(result_map[i]) / sum(result_map.values())
        elif i == PROTO_DICT['gnutella'] and sum(result_map.values()) != 0:
            st_gnutella = float(result_map[i]) / sum(result_map.values())
        elif i == PROTO_DICT['edonkey'] and sum(result_map.values()) != 0:
            st_edonkey = float(result_map[i]) / sum(result_map.values())
        elif i == PROTO_DICT['bittorrent'] and sum(result_map.values()) != 0:
            st_bittorrent = float(result_map[i]) / sum(result_map.values())
        elif i == PROTO_DICT['skype'] and sum(result_map.values()) != 0:
            st_skype = float(result_map[i]) / sum(result_map.values())
    if DEBUG:
        print(st.format('Cluster' + str(cluster_number), st_http, st_gnutella, st_edonkey, st_bittorrent))
    train_flow_result = TrainFlowResult()
    st = ''
    train_flow_result.group = st
    train_flow_result.probability_for_http = st_http
    train_flow_result.probability_for_gnutella = st_gnutella
    train_flow_result.probability_for_edonkey = st_edonkey
    train_flow_result.probability_for_bittorrent = st_bittorrent
    train_flow_result.probability_for_skype = st_skype
    train_flow_result.number_of_flows = sum(result_map.values())
    train_flow_result.calculate_population_fraction()
    train_flow_result.calculate_final_probability()
    return train_flow_result

def get_indices_for_pred(pred, cluter_number):
    return np.where(pred == cluter_number)

def combine_labels_data(data, labels):
    return np.hstack((data, labels))

def testing(test_data, test_labels, k_means, train_result_map):
    index = 0
    true_positive = 0
    total_flows = 0
    for test_data_row in test_data:
        pred = k_means.predict(test_data_row)
        if PROTO_DICT_MAP[int(train_result_map[int(pred)].application)] == PROTO_DICT_MAP[int(test_labels[index, -1])]:
            true_positive += 1
        total_flows += 1
        index += 1
    return float(true_positive)/float(total_flows)
    
def create_plots(plot_map, plot_selected_map):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_yticks(np.arange(0.6,1,0.05))
    plt.plot(plot_map.keys(), plot_map.values(), 'r*', label="K-Means All Attrs")
    plt.plot(plot_selected_map.keys(), plot_selected_map.values(), 'g^', label="K-Means With Selected")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.6, 1.0])
    plt.grid()
    plt.legend(loc='best')
    plt.title('Percent of Training v/s Accuracy')
    plt.xlabel('Percent of Data')
    plt.ylabel('Accuracy')
    plt.show() 

def get_attribute_group_final(data, attribute_list):
    return data[attribute_list]

def main():
    input_data = get_numpy_array_from_file("mofifiedinput.csv")
    train_data, train_labels, test_data, test_labels = return_train_test_labels(input_data, None)
    plot_with_all_map = {}
    plot_with_selected_map = {}
    number_of_iters = 5
    for clus in range(10, 150, 10):
        final_accu = 0.0
        for i in range(number_of_iters):
            k_means = KMeans(init='random', n_clusters=clus, n_init=50)
            k_means.fit(train_data)
            pred = k_means.labels_
            train_result_map = {}
            for i in range(clus):
                indices = get_indices_for_pred(pred, i)
                refined_data = np.copy(train_data)
                refined_data = combine_labels_data(refined_data, train_labels)
                refined_data = refined_data[indices, -1]
                train_result_map[i] = generate_stats_for_cluster(refined_data, i)
            final_accu += testing(test_data, test_labels, k_means, train_result_map)
            print(final_accu)
        plot_with_all_map[clus] = float(final_accu)/number_of_iters
        print('Final Accuracy: '+str(plot_with_all_map[clus])) 
        
        final_accu = 0.0
        for i in range(number_of_iters):
            k_means = KMeans(init='random', n_clusters=clus, n_init=50)
            train_copy = np.copy(train_data)
            test_copy = np.copy(test_data)
            train_copy = get_attribute_group_modified(train_copy, [7, 8, 9, 10, 11, 13, 16])
            test_copy = get_attribute_group_modified(test_copy, [7, 8, 9, 10, 11, 13, 16])
            k_means.fit(train_copy)
            pred = k_means.labels_
            train_result_map = {}
            for i in range(clus):
                indices = get_indices_for_pred(pred, i)
                refined_data = np.copy(train_data)
                refined_data = combine_labels_data(refined_data, train_labels)
                refined_data = refined_data[indices, -1]
                train_result_map[i] = generate_stats_for_cluster(refined_data, i)
            final_accu += testing(test_copy, test_labels, k_means, train_result_map)
            print(final_accu)
        plot_with_selected_map[clus] = float(final_accu)/number_of_iters
        print('Final selected Accuracy: '+str(plot_with_selected_map[clus]))    
    create_plots(plot_with_all_map, plot_with_selected_map)

if __name__ == '__main__':
    main()