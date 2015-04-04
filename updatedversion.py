__author__ = 'KattaAnil'


import numpy as np
from sklearn.cluster import KMeans
import csv
import itertools

PROTO_DICT_MAP = {0: 'http', 1: 'gnutella', 2: "edonkey", 3: "bittorrent", 4: "skype"}

PROTO_DICT = {'http': 0.0, 'gnutella': 1.0, "edonkey": 2.0, "bittorrent": 3.0, "skype":4.0}

DEBUG = False

ATTRIBUTE_MAP = {0:'Avg IAT', 1: "Min IAT", 2:"Max IAT", 3:"Std Div IAT",
                 4:"Avg RIAT", 5:"Max RIAT", 6:"Std Div RIAT",
                 7:'Avg Pkt Size', 8:'Min Pkt Size', 9:'Max Pkt Size', 10:'Std Div Pkt Size',
                 11:"Total Pkt Size", 12:"Flow Duration", 13:"Number of Packets", 14:"Avg Pkts Per Sec",
                 15:"Avg Bytes Per Sec", 16:"Payload Size"} 


ATTRIBUTE_GROUP_LIST = ['pkt size', 'flow']

#ATTRIBUTE_GROUP_LIST = ['pkt size']
def return_permutations_of_group(group_name):
    if group_name.lower() == "pkt size":
        return [[7, 10], [7, 8, 9]]
    elif group_name.lower() == "flow":
        return [[11, 16], [13, 16]]



def get_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")


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

def get_attribute_group_final(data, attribute_list):
    return data[attribute_list]


def get_indices_for_pred(pred, cluter_number):
    return np.where(pred == cluter_number)


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



class TesFlowResult(object):

    def __init__(self):
        self.testing_flow = None
        self.actual_protocol = None
        self.classification_result = None
        self.probability = None
        self.true_positive = None
        self.group_pkt_size_one_population = None
        self.group_pkt_size_two_population = None
        self.group_flow_one_population = None
        self.group_flow_two_population = None
        self.avg_population_fraction = None
        self.simple_probability = None
        self.number_of_flows_train = None
        self.considered_fraction = None
        self.group_one_label = None
        self.group_two_label = None
        self.group_three_label = None
        self.group_four_label = None

    def write_to_csv(self, csv_file):
        fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Considered Fraction', 
                      'Group 1 Fraction', 'Group 2 Fraction', 'Group 3 Fraction', 'Group 4 Fraction', 
                      'Group 1 label', 'Group 2 label', 'Group 3 label', 'Group 4 label', 'Number of Flows', 
                      'True Positive']

        csv_file.writerow({'Testing Flow':self.testing_flow, 'Actual Protocol':self.actual_protocol , 
                           'Classification Result': self.classification_result, 'Considered Fraction':self.considered_fraction,
                           'Group 1 Fraction':self.group_flow_one_population, 'Group 2 Fraction':self.group_flow_two_population,
                           'Group 3 Fraction':self.group_flow_three_population, 'Group 4 Fraction':self.group_flow_four_population,
                           'Group 1 label':self.group_one_label, 'Group 2 label':self.group_two_label, 'Group 3 label':self.group_three_label,
                           'Group 4 label':self.group_four_label, 'Number of Flows': self.number_of_flows_train, 
                           'True Positive':self.true_positive})

    def calculate_avg_population_fraction(self):
        total = 0.0
        if self.group_flow_one_population:
            total += self.group_flow_one_population
        if self.group_flow_three_population:
            total += self.group_flow_three_population
        if self.group_flow_two_population:
            total += self.group_flow_two_population
        if self.group_flow_four_population:
            total += self.group_flow_four_population
        self.avg_population_fraction = float(total) / 4.0


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
    for i in attribute_group:
        st += ATTRIBUTE_MAP[i] + ' '
    st = st[:len(st)]
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

final_considered_map = {}

k_means_map = {}

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
    k_means_map[get_app_string(attribute_group)] = k_means_cls
    final_considered_map[get_app_string(attribute_group)] = train_result_map
    #Testing Begins here
    return (k_means_cls, train_result_map)

import random



OLD_CODE = False

def testing(test_data, test_labels, consider_all, number_of_clusters):
    if DEBUG:
        print('Test Rows: ' + str(test_data.shape[0]))
    true_positive = 0
    index = 0
    attr_list = []
    fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Considered Fraction', 
                      'Group 1 Fraction', 'Group 2 Fraction', 'Group 3 Fraction', 'Group 4 Fraction', 
                      'Group 1 label', 'Group 2 label', 'Group 3 label', 'Group 4 label', 'Number of Flows', 
                      'True Positive']
    output_file = open('./results/'+SELECTION_ALGO +'-clusters-'+str(number_of_clusters) + '.csv   ', 'w', 20)
    print('Output file has been created '+str(output_file))
    csv_file = csv.DictWriter(output_file, delimiter=",", fieldnames=fieldnames)
    csv_file.writeheader()
    for att in ATTRIBUTE_GROUP_LIST:
        for sub in return_permutations_of_group(att):
            attr_list.append(sub)
    for test_data_row in test_data:
        temp_result_temp = {}
        test_result = TesFlowResult()
        for att in attr_list:
            after_attr_set = get_attribute_group_final(test_data_row, att)
            k_means_for_this = k_means_map[get_app_string(att)]
            if DEBUG:
                print(after_attr_set)
            pred = k_means_for_this.predict(after_attr_set)
            train_result_map = final_considered_map[get_app_string(att)]
            if DEBUG:
                print('Pred: ' + str(pred))
                print('Att: ' + str(att))
                print('Result: ' + str(train_result_map[int(pred)]))
            if att == [7, 10]:
                test_result.group_flow_one_population = train_result_map[int(pred)].population_fraction
                temp_result_temp[1] = train_result_map[int(pred)]
                test_result.group_one_label = PROTO_DICT_MAP[train_result_map[int(pred)].application]
            elif att == [7, 8 , 9]:
                test_result.group_flow_two_population = train_result_map[int(pred)].population_fraction
                temp_result_temp[2] = train_result_map[int(pred)]
                test_result.group_two_label = PROTO_DICT_MAP[train_result_map[int(pred)].application]
            elif att == [11, 16]:
                test_result.group_flow_three_population = train_result_map[int(pred)].population_fraction
                temp_result_temp[3] = train_result_map[int(pred)]
                test_result.group_three_label = PROTO_DICT_MAP[train_result_map[int(pred)].application]
            elif att == [13, 16]:
                test_result.group_flow_four_population = train_result_map[int(pred)].population_fraction
                temp_result_temp[4] = train_result_map[int(pred)]
                test_result.group_four_label = PROTO_DICT_MAP[train_result_map[int(pred)].application]

        test_result.calculate_avg_population_fraction()
        test_result.number_of_flows_train = train_result_map[int(pred)].number_of_flows
        test_result.actual_protocol = PROTO_DICT_MAP[int(test_labels[index, :][0])]

        if SELECTION_ALGO == 'UNANIMOUS-GREATEST':
            result = True
            temp = None
            for ikey in temp_result_temp.keys():
                temp = temp_result_temp[ikey]
                break
            for ikey in temp_result_temp.keys():
                if temp.application != temp_result_temp[ikey].application:
                    result = False
            if result:
                test_result.classification_result = PROTO_DICT_MAP[temp.application]
                test_result.considered_fraction = temp.population_fraction
            else:
                max_value = 0.0
                max_train_result = None
                for ikey in temp_result_temp.keys():
                    if temp_result_temp[ikey].population_fraction > max_value:
                        max_value = temp_result_temp[ikey].population_fraction
                        max_train_result = temp_result_temp[ikey]
                test_result.classification_result = PROTO_DICT_MAP[max_train_result.application]
                test_result.considered_fraction = max_train_result.population_fraction

        if SELECTION_ALGO == 'UNANIMOUS-QUORUM':
            result = True
            temp = None
            for ikey in temp_result_temp.keys():
                temp = temp_result_temp[ikey]
                break
            for ikey in temp_result_temp.keys():
                if temp.application != temp_result_temp[ikey].application:
                    result = False
            if result:
                test_result.classification_result = PROTO_DICT_MAP[temp.application]
                test_result.considered_fraction = temp.population_fraction
            else:
                counter_map = {}
                train_temp_result_map = {}
                for ikey in temp_result_temp.keys():
                    if temp_result_temp[ikey].application in counter_map:
                        counter_map[temp_result_temp[ikey].application] += 1
                    else:
                        counter_map[temp_result_temp[ikey].application] = 1
                        train_temp_result_map[temp_result_temp[ikey].application] = temp_result_temp[ikey]
                max_counter = max(counter_map.values())
                temp_train_result = None
                for ikey in counter_map.keys():
                    if counter_map[ikey] == max_counter:
                        temp_train_result = train_temp_result_map[ikey]            
                test_result.classification_result = PROTO_DICT_MAP[temp_train_result.application]
                test_result.considered_fraction = temp_train_result.population_fraction

        if SELECTION_ALGO == 'UNANIMOUS-RANDOM':
            result = True
            temp = None
            for ikey in temp_result_temp.keys():
                temp = temp_result_temp[ikey]
                break
            for ikey in temp_result_temp.keys():
                if temp.application != temp_result_temp[ikey].application:
                    result = False
            if result:
                test_result.classification_result = PROTO_DICT_MAP[temp.application]
                test_result.considered_fraction = temp.population_fraction
            else:
                ranint = random.randint(1, 4)
                train_result_temp = temp_result_temp[ranint]
                test_result.classification_result = PROTO_DICT_MAP[train_result_temp.application]
                test_result.considered_fraction = train_result_temp.population_fraction

        if OLD_CODE:            
            if test_result.group_flow_one_population > THRESHOLD_FRACTION:
                test_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(pred)].application)]
                test_result.considered_fraction = test_result.group_flow_one_population
            elif test_result.group_flow_two_population > THRESHOLD_FRACTION:
                test_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(pred)].application)]
                test_result.considered_fraction = test_result.group_flow_two_population
            elif test_result.group_flow_three_population > THRESHOLD_FRACTION:
                test_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(pred)].application)]
                test_result.considered_fraction = test_result.group_flow_three_population
            elif test_result.group_flow_four_population > THRESHOLD_FRACTION:
                test_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(pred)].application)]
                test_result.considered_fraction = test_result.group_flow_four_population
            else:
                test_result.classification_result = "unknown"
                test_result.considered_fraction = -1.0
        test_result.testing_flow = index
        index += 1
        if test_result.actual_protocol == test_result.classification_result:
            true_positive += 1
            test_result.true_positive = True
        else:
            test_result.true_positive = False 
        test_result.write_to_csv(csv_file)
    return float(true_positive) / index


def generate_all_possible_combinations_of_list(input_list):
    return_list = []
    for length in range(1, len(input_list) + 1):
        for sub in itertools.combinations(input_list, length):
            return_list.append(list(sub))
    return return_list

def get_app_string(input_list):
    return_string = ''
    for i in input_list:
        return_string += ATTRIBUTE_MAP[i] + ' '
    return return_string[:len(return_string) - 1]


THRESHOLD_FRACTION = 0.6

SELECTION_ALGO = 'RANDOM'
SELECTION_ALGO_LIST = ['RANDOM', 'GREATEST', 'QUORUM', 'UNANIMOUS-GREATEST', 'UNANIMOUS-QUORUM', 'UNANIMOUS-RANDOM']
SELECTION_ALGO_LIST = ['UNANIMOUS-GREATEST']
plot_map = {}
for i in SELECTION_ALGO_LIST:
    plot_map[i] = {}


supervised_plot_map = {}
unsupervised_plot_map = {}

def main_modified(input_data, outputfile, number_of_clusters=20, number_of_times=10, consider_all=True, number_of_repetetions=10, percent=0.9):
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None, percent=percent)
    print('Train: '+str(train.shape))
    print('Test: '+str(test.shape))
    if DEBUG:
        print(train.shape)
    fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
    print('After reading testing and training datasets')
    for clus in range(10, number_of_clusters, 10):
        pass
    clus = 40
    for att in ATTRIBUTE_GROUP_LIST:
        attribute_list = return_permutations_of_group(att)
        for sub in attribute_list:
            final_accu = 0.0
            after_attribute_set = get_attribute_group_modified(train, sub)
            after_attribute_test_set = get_attribute_group_modified(test, sub)
            if DEBUG:
                print(after_attribute_set.shape)
                print(after_attribute_test_set.shape)
            for i in range(number_of_repetetions):
                with open('./results/resultsfinalgroup' + str(get_app_string(sub)) + '.csv', 'w', 20) as output_file:
                    csv_file = csv.DictWriter(output_file, delimiter=",", fieldnames=fieldnames)
                    csv_file.writeheader()
                    if DEBUG:
                        print('{0:15}{1:10}{2:10}{3:10}{4:10}'.format('Cluster', 'Http', 'Gnutella', 'Edonkey', 'Bittorrent'))                
                    k_means_cls, train_result_map = k_means(after_attribute_set, train_labels, clus, number_of_times, sub) 

    for se in SELECTION_ALGO_LIST:
        global SELECTION_ALGO
        SELECTION_ALGO = se
        final_accu = 0.0
        for i in range(number_of_repetetions):
            return testing(test, test_labels, consider_all, clus)
            final_accu += testing(test, test_labels, consider_all, clus)
            print('Final Acc: ' + str(final_accu))        
        #plot_map[SELECTION_ALGO][clus] = float(final_accu)/ number_of_repetetions
        print('Final Accuracy: ' + str(SELECTION_ALGO) + ' ' + str(final_accu / number_of_repetetions))    
    print('Done processing!')         

import matplotlib.pyplot as plt
from matplotlib.pylab import figure
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

def supervised(input_data, percent=0.9):
    print("Inside Supervised")
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None, percent=percent)
    print('Train: '+str(train.shape))
    print('Test: '+str(test.shape))
    train = get_attribute_group_modified(train, [7, 8, 9, 10, 11, 13, 16])
    test = get_attribute_group_modified(test, [7, 8, 9, 10, 11, 13, 16])
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    supervised_plot_map[percent] = accuracy_score(test_labels, pred)
    print(accuracy_score(test_labels, pred))

def create_plots():
    print(plot_map)
    print('Inside Create Plots')
    nrows = len(SELECTION_ALGO_LIST) + 1 
    index = 1   
    print('Number of Rows: '+str(nrows))
    
    for se in plot_map.keys():
        print('SE')
        figure(index)
        plt.plot(plot_map[se].keys(), plot_map[se].values(), 'o', label=se)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Clusters')
        plt.title('Number of Clusters v/s Accuracy for '+str(se))
        index += 1
    print('Plotting first few plots are done: '+str(index))
    figure(index)
    #plt.subplot(nrows, 1, index)
    for se in plot_map.keys():
        plt.plot(plot_map[se].keys(), plot_map[se].values(), label=se)
    plt.title('Number of Clusters v/s Accuracy of Various Algos')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.show()

def create_comparision_plots():
    print(unsupervised_plot_map)
    print(supervised_plot_map)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_yticks(np.arange(0.6,1,0.05))
    plt.plot(unsupervised_plot_map.keys(), unsupervised_plot_map.values(), 'r*', label="Hypothesis")
    plt.plot(supervised_plot_map.keys(), supervised_plot_map.values(), 'g^', label='Supervised')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.6, 1.0])
    plt.grid()
    plt.legend(loc='best')
    plt.title('Percent of Training v/s Accuracy')
    plt.xlabel('Percent of Data')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    print("New New updated version of the file")
    inp = 1
    consider_all = True
    if inp == 1:
        consider_all = False
    else:
        consider_all = True
    output_file = open("resultscompiled.txt", "w", 20)
    input_data = get_numpy_array_from_file("mofifiedinput.csv")
    number_of_iterations = 5
    percentage_range = 20
    for i in range(1, percentage_range):
        percent = i/float(percentage_range)
        final_accu = 0.0
        for i in range(number_of_iterations):
            final_accu += main_modified(input_data, output_file, 150, 50, consider_all, 5, percent=percent)
            print('Final Accu: '+str(final_accu))
        unsupervised_plot_map[percent] = float(final_accu)/number_of_iterations
        supervised(input_data, percent=percent)
    print("Done with Calculations")
    create_comparision_plots()
    output_file.close()
