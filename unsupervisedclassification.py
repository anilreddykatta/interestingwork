__author__ = 'KattaAnil'


import numpy as np
from sklearn.cluster import KMeans
from TrainFlowResult import TrainFlowResult
from TestFlowResult import TesFlowResult
import csv
import matplotlib.pyplot as plt
import itertools

PROTO_DICT_MAP = {0: 'http', 1: 'gnutella', 2: "edonkey", 3: "bittorrent"}

PROTO_DICT = {'http': 0.0, 'gnutella': 1.0, "edonkey": 2.0, "bittorrent": 3.0}

DEBUG = False

ATTRIBUTE_MAP = {0:'Avg IAT', 1: "Min IAT", 2:"Max IAT", 3:"Std Div IAT",
                 4:"Avg RIAT", 5:"Max RIAT", 6:"Std Div RIAT",
                 7:'Avg Pkt Size', 8:'Min Pkt Size', 9:'Max Pkt Size', 10:'Std Div Pkt Size',
                 11:"Total Pkt Size", 12:"Flow Duration", 13:"Number of Packets", 14:"Avg Pkts Per Sec",
                 15:"Avg Bytes Per Sec", 16:"Payload Size"}


ATTRIBUTE_GROUP_LIST = ['iat', 'riat', 'pkt size', 'flow']

#ATTRIBUTE_GROUP_LIST = ['pkt size']

def return_permutations_of_group(group_name):
    if group_name.lower() == "iat":
        return generate_all_possible_combinations_of_list([0, 1, 2, 3])
    elif group_name.lower() == 'riat':
        return generate_all_possible_combinations_of_list([4, 5, 6])
    elif group_name.lower() == 'pkt size':
        return generate_all_possible_combinations_of_list([7, 8, 9, 10])
    elif group_name.lower() == 'flow':
        return generate_all_possible_combinations_of_list([11, 12, 13, 14, 15, 16])



def get_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")


def return_train_test_labels(input_data, number_of_each_flows, percent=0.9):
    input_data = np.copy(input_data)
    http_data = input_data[input_data[:, 17] == PROTO_DICT['http']]
    gnutella_data = input_data[input_data[:, 17] == PROTO_DICT['gnutella']]
    edonkey_data = input_data[input_data[:, 17] == PROTO_DICT['edonkey']]
    bittorrent_data = input_data[input_data[:, 17] == PROTO_DICT['bittorrent']]

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


def combine_labels_data(data, labels):
    return np.hstack((data, labels))


def get_attribute_group(data, attribute_group=1):
    if attribute_group == 1:
        return data[:, :4]
    elif attribute_group == 2:
        return data[:, 4:8]
    elif attribute_group == 3:
        return data[:, 8:]


def get_attribute_group_modified(data, attribute_list):
    if len(attribute_list) == 1:
        return_data = data[:,attribute_list]
        return_data = return_data.reshape((return_data.shape[0], 1))
        return return_data
    else:
        return_data = data[:, attribute_list]
        return return_data


def get_indices_for_pred(pred, cluter_number):
    return np.where(pred == cluter_number)


def generate_stats_for_cluster(data, cluster_number, attribute_group):
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
    for i in result_map:
        if i == PROTO_DICT['http'] and sum(result_map.values()) != 0:
             st_http = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['gnutella'] and sum(result_map.values()) != 0:
            st_gnutella = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['edonkey'] and sum(result_map.values()) != 0:
            st_edonkey = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['bittorrent'] and sum(result_map.values()) != 0:
            st_bittorrent = float(result_map[i])/sum(result_map.values())
    if DEBUG:
        print(st.format('Cluster'+str(cluster_number), st_http, st_gnutella, st_edonkey, st_bittorrent))
    train_flow_result = TrainFlowResult()
    st = ''
    for i in attribute_group:
        st += ATTRIBUTE_MAP[i] + ' '
    st = st[:len(st)]
    train_flow_result.group = st
    train_flow_result.probability_for_http = st_http
    train_flow_result.probability_for_gnutella = st_gnutella
    train_flow_result.probability_for_edonkey = st_edonkey
    train_flow_result.probability_for_bittorrent = st_bittorrent
    train_flow_result.calculate_final_probability()
    return train_flow_result


def k_means(data, train_labels, num_clusters, number_of_times, attribute_group):
    k_means_cls = KMeans(init='random', n_clusters=num_clusters, n_init=number_of_times)
    k_means_cls.fit(data)
    pred = k_means_cls.labels_
    train_result_map = {}
    for i in range(num_clusters):
        indices = get_indices_for_pred(pred, i)
        refined_data = np.copy(data)
        refined_data = combine_labels_data(refined_data, train_labels)
        refined_data = refined_data[indices, -1]
        train_result_map[i] = generate_stats_for_cluster(refined_data, i, attribute_group)

    #Testing Begins here
    return (k_means_cls, train_result_map)

def testing(test_data, test_labels, k_means_cls, train_result_map, attribute_group, csv_file, consider_all):
    pred = k_means_cls.predict(test_data)
    index = 0
    test_flow_result = None
    true_positive = 0
    results_map = {}
    for i in pred.tolist():
        test_flow_result = TesFlowResult()
        test_flow_result.actual_protocol = PROTO_DICT_MAP[int(test_labels[index, :][0])]
        test_flow_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(i)].application)]
        if consider_all and train_result_map[int(i)].final_probability < 0.8:
            continue
        test_flow_result.probability = train_result_map[int(i)].final_probability
        test_flow_result.testing_flow = index
        test_flow_result.true_positive = (PROTO_DICT_MAP[test_labels[index, :][0]] == PROTO_DICT_MAP[int(train_result_map[int(i)].application)])
        if test_flow_result.true_positive:
            true_positive += 1
        test_flow_result.write_to_csv(csv_file)
        index += 1
        if test_flow_result.classification_result:
            results_map[test_flow_result.probability] = 2.0
        else:
            results_map[test_flow_result.probability] = 1.0
    if DEBUG:
        pass
    print('Temp Accuracy Iter 1: '+str(float(true_positive)/index))
    return float(true_positive)/index


def generate_all_possible_combinations_of_list(input_list):
    return_list = []
    for length in range(1, len(input_list) + 1):
        for sub in itertools.combinations(input_list, length):
            return_list.append(list(sub))
    return return_list


def main(number_of_clusters=20, number_of_times=10, consider_all = True):
    input_data = get_numpy_array_from_file("mofifiedinput.csv")
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None)
    fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
    for i in range(1, 4):
        if i != 2:
            continue
        with open('resultsfinalgroup'+str(i)+'.csv', 'w', 20) as output_file:
            csv_file = csv.DictWriter(output_file, delimiter=",", fieldnames=fieldnames)
            csv_file.writeheader()
            if DEBUG:
                print('{0:15}{1:10}{2:10}{3:10}{4:10}'.format('Cluster', 'Http', 'Gnutella', 'Edonkey', 'Bittorrent'))
            after_attribute_set = get_attribute_group(train, i)
            k_means_cls, train_result_map = k_means(after_attribute_set, train_labels, number_of_clusters, number_of_times, i)
            after_attribute_test_set = get_attribute_group(test, i)
            final_accuracy = testing(after_attribute_test_set, test_labels, k_means_cls, train_result_map, i, csv_file, consider_all)
            print('='*15+str('Atrribute Group '+str(i))+'='*15)


def get_app_string(input_list):
    return_string = ''
    for i in input_list:
        return_string += ATTRIBUTE_MAP[i] + ' '
    return return_string


def main_modified(attribute_list, outputfile, number_of_clusters=20, number_of_times=10, consider_all=True, number_of_repetetions=10):
    input_data = get_numpy_array_from_file("mofifiedinput.csv")
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None)
    print(train.shape)
    fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
    for sub in attribute_list:
        final_accu = 0.0
        after_attribute_set = get_attribute_group_modified(train, sub)
        after_attribute_test_set = get_attribute_group_modified(test, sub)
        if DEBUG:
            print(after_attribute_set.shape)
            print(after_attribute_test_set.shape)
        for i in range(number_of_repetetions):
            with open('resultsfinalgroup'+str(get_app_string(sub))+'.csv', 'w', 20) as output_file:
                csv_file = csv.DictWriter(output_file, delimiter=",", fieldnames=fieldnames)
                csv_file.writeheader()
                if DEBUG:
                    print('{0:15}{1:10}{2:10}{3:10}{4:10}'.format('Cluster', 'Http', 'Gnutella', 'Edonkey', 'Bittorrent'))                
                k_means_cls, train_result_map = k_means(after_attribute_set, train_labels, number_of_clusters, number_of_times, sub)                
                final_accuracy = testing(after_attribute_test_set, test_labels, k_means_cls, train_result_map, sub, csv_file, consider_all)
                final_accu += final_accuracy
        print("Final Accuracy: "+str(final_accu/number_of_repetetions))
        print('='*15+str('Atrribute Group '+str(get_app_string(sub)))+'='*15)
        outputfile.write("Final Accuracy: "+str(final_accu/number_of_repetetions)+"\n")
        outputfile.write('-'*15+str('Atrribute Group '+str(get_app_string(sub)))+'-'*15+"\n")


if __name__ == '__main__':
    #inp = int(input("Enter 1 or 0 whether to use all or only > 0.8: "))
    inp = 1
    consider_all = True
    if inp == 1:
        consider_all = False
    else:
        consider_all = True
    output_file = open("resultscompiled.txt", "w", 20)
    #with open("resultscompiled.txt", "w", 20) as output_file:
    for att in ATTRIBUTE_GROUP_LIST:
        attr_list = return_permutations_of_group(att)
        main_modified(attr_list, output_file, 25, 50, consider_all, 3)
    output_file.close()