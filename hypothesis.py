#!/usr/bin/env python
__author__ = 'KattaAnil'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ClusteringClassificationModel
import matplotlib.pyplot as plt
import operator

INPUT_FILE = "C:\\Users\\KattaAnil\\Desktop\\DR_KIM_RA\\FreshCopy\\realinput.csv"

#OUTPUT_FILE = "C:\Users\KattaAnil\Desktop\DR_KIM_RA\New Work\Python Scripts\output_latest.csv"

OUTPUT_FILE = 'realinput.csv'

PROTO_DICT = {'http':0, 'gnutella':1, "edonkey":2, "bittorrent":3}

PROTO_DICT_MAP = {0:'http', 1:'gnutella', 2:"edonkey", 3:"bittorrent"}

PURITY_OF_CLUSTER = 0.90

MINIMUM_FLOWS_IN_CLUSTER = 2

ATTRIBUTE_MAP = {0:"avg_iat", 1:"min_iat", 2:"max_iat", 3:"std_div_iat", 4:"avg_pkt_size",
                 5:"min_pkt_size", 6:"max_pkt_size", 7:"std_div_pkt_size", 8:"total_pkt_size",
                 9:"flow_duration", 10:"number_of_packets", 11:"avg_pkts_per_sec", 12:"avg_bytes_per_sec",
                 13:"payload_size", 14:"protocol"}


classification_model_list = []

def format_input_file():
    with open(INPUT_FILE) as input_file, open(OUTPUT_FILE, 'w') as output_file:
        next(input_file)
        for line in input_file:
            output_file.write(return_new_line(line))


def return_new_line(old_line):
    line_array = old_line.strip().split(",")
    line_array[14] = PROTO_DICT[line_array[14]]
    return_string = ",".join(map(str, line_array))
    return_string += "\n"
    return return_string

def when_to_stop(data, pred, cluster_number, property_used):
    nrows, ncols = data.shape
    index_with_cluster_number = np.where(pred == cluster_number)[0]
    apps = {}
    for ind in index_with_cluster_number:
        if data[ind][-1] in apps:
            apps[data[ind][-1]] += 1
        else:
            apps[data[ind][-1]] = 1
    if len(apps) == 0:
        return (-1, 0.0)

    a = float(max(apps.values()))/sum(apps.values())
    if float(max(apps.values()))/sum(apps.values()) >= PURITY_OF_CLUSTER and sum(apps.values()) >= MINIMUM_FLOWS_IN_CLUSTER:
        return (max(apps.keys()), float(max(apps.values()))/sum(apps.values()))
    else:
        return (-1, 0.0)

def cluster_decisions(data, pred, property_used):
    number_of_clusters = np.amax(pred)
    further_clusters_to_be_retained = []
    accuracy_map = {}
    cluster_app_map = {}
    final_cluster_app_map = {}
    for i in range(number_of_clusters + 1):
        app, accuracy = when_to_stop(data, pred, i, property_used)
        if app >= 0:
            cluster_app_map[i] = app
            accuracy_map[i] = accuracy
        else:
            accuracy_map[i] = 0.0
    for ke in accuracy_map.keys():
        #if (accuracy_map[ke] < max(accuracy_map.values()) and accuracy_map[ke] < PURITY_OF_CLUSTER) or accuracy_map[ke] == 0.0:
        if accuracy_map[ke] < PURITY_OF_CLUSTER or accuracy_map[ke] == 0.0:
            further_clusters_to_be_retained.append(ke)
        else:
            final_cluster_app_map[ke] = cluster_app_map[ke]
    return (final_cluster_app_map, further_clusters_to_be_retained)

def indices_to_be_retained_for_next_iteration(data, pred, property_used):
    final_cluster_app_map, clusters_to_be_retained = cluster_decisions(data, pred, property_used)
    if len(clusters_to_be_retained) > 0:
        indices = []
        for clus in clusters_to_be_retained:
            for ind in np.where(pred == clus)[0]:
                indices.append(ind)
        return (indices, clusters_to_be_retained, final_cluster_app_map)

def refine_data_for_next_iteration(data, indices):
    if indices:
        new_data = np.array([])
        for ind in indices:
            if new_data.size > 0:
                new_data = np.vstack((new_data, data[ind]))
            else:
                new_data = data[ind]
        return new_data
    else:
        return data

def build_classification_model(number_of_clusters, clusters_to_be_retained, classification_model, final_cluster_app_map, attribute):
    global classification_model_list
    cluster_classification_model = ClusteringClassificationModel.\
                ClusterClassificationModel(iter_iter, classification_model, attribute)
    for clus in number_of_clusters:
        if clus in clusters_to_be_retained:
            pass
        else:
            cluster_classification_model.application_map[clus] = final_cluster_app_map[clus]
    classification_model_list.append(cluster_classification_model)

def run_for_every_attribute_return_best_one(data):
    kmeans = KMeans(init='random', n_clusters=10, n_init=5)
    nrows, ncols = data.shape
    max_indices_to_be_removed = 0
    max_indices_to_be_removed_list = []
    best_attribute = None
    for i in range(ncols - 1):
        input_data = data[:, i]
        input_data = input_data.reshape((input_data.shape[0], 1))
        kmeans.fit(input_data)
        pred = kmeans.predict(input_data)
        number_of_clusters = np.amax(pred)
        number_of_indices_to_be_removed, clusters_to_be_retained, final_cluster_app_map = indices_to_be_retained_for_next_iteration(data, pred, i)
        number_of_clusters = list(range(number_of_clusters + 1))
        build_classification_model(number_of_clusters, clusters_to_be_retained, kmeans, final_cluster_app_map, i)
        if number_of_indices_to_be_removed and len(number_of_indices_to_be_removed) > max_indices_to_be_removed:
            best_attribute = i
            max_indices_to_be_removed = len(number_of_indices_to_be_removed)
            max_indices_to_be_removed_list = number_of_indices_to_be_removed
    data = data[max_indices_to_be_removed_list, :]
    return (data, best_attribute)

output_file = open("results_imp.txt", "w", 0)
MAXIMUM_CLUSTER_SIZE = 100

iter_iter = 0
def k_means(data):
    nrows, ncols = data.shape
    for i in range(ncols - 1):
        global iter_iter
        iter_iter += 1
        data, attribute = run_for_every_attribute_return_best_one(data)
        if attribute:
            output_file.write("For Iter: "+str(iter_iter)+" Best Attribute is: "+ATTRIBUTE_MAP[attribute]+"\n")
        else:
            return
    nrows, ncols = data.shape
    if nrows >= MAXIMUM_CLUSTER_SIZE:
        return k_means(data)
    else:
        return

def get_data(data, number_of_flows=100):
    input_data = np.copy(data)
    http_data = input_data[input_data[:, 14] == PROTO_DICT['http']]
    http_data = http_data[:number_of_flows, :]
    gnutella_data = input_data[input_data[:, 14] == PROTO_DICT['gnutella']]
    gnutella_data = gnutella_data[:number_of_flows, :]
    edonkey_data = input_data[input_data[:, 14] == PROTO_DICT['edonkey']]
    edonkey_data = edonkey_data[:number_of_flows, :]
    bittorrent_data = input_data[input_data[:, 14] == PROTO_DICT['bittorrent']]
    bittorrent_data = bittorrent_data[:number_of_flows, :]
    input_data = np.vstack((http_data, gnutella_data, edonkey_data, bittorrent_data))
    return input_data

def test_classification_model(test_data, percent, output_file):
    #output_file.write('Percent of Training Data is: '+str(percent)+'\n')
    nrows, ncols = test_data.shape
    http_test = test_data[test_data[:, 14] == PROTO_DICT['http']]
    gnutella_test = test_data[test_data[:, 14] == PROTO_DICT['gnutella']]#np.where(test_data[:, 14] == PROTO_DICT['gnutella'])
    edonkey_test = test_data[test_data[:, 14] == PROTO_DICT['edonkey']]#np.where(test_data[:, 14] == PROTO_DICT['edonkey'])
    bittorrent_test = test_data[test_data[:, 14] == PROTO_DICT['bittorrent']]#np.where(test_data[:, 14] == PROTO_DICT['bittorrent'])
    print('Http: '+str(http_test.shape))
    print('Gnutella: '+str(gnutella_test.shape))
    print('Bittorrent: '+str(edonkey_test.shape))
    print('eDonkey: '+str(bittorrent_test.shape))
    total_flows = nrows
    #output_file.write('Total Number of Flows: '+str(total_flows)+'\n')
    true_positive = 0
    http_tp = 0
    edonkey_tp = 0
    gnutella_tp = 0
    bittorrent_tp = 0

    false_positive = 0
    http_fp = 0
    edonkey_fp = 0
    gnutella_fp = 0
    bittorrent_fp = 0
    for i in range(nrows):
        for model in classification_model_list:
            data = test_data[i, model.attribute]
            result = model.model.predict(data)
           # print('Expected: '+str(PROTO_DICT_MAP[model.application])+' Actual: '+str(PROTO_DICT_MAP[int(test_data[i, -1])])+' Cluster predicted: '+str(model.model)+'\n')
            #output_file.write('Expected: '+str(PROTO_DICT_MAP[model.application])+' Actual: '+str(PROTO_DICT_MAP[int(test_data[i, -1])])+' Cluster predicted: '+str(model)+'\n');
            #if result in model.clusters_with_high_purity and float(test_data[i, -1]) == model.application:
            take_break = False
            for cl in model.application_map.keys():
                if result == cl:
                    if float(test_data[i, -1]) == model.application_map[cl]:
                        true_positive += 1
                        if model.application_map[cl] == PROTO_DICT['http']:
                            http_tp += 1
                        elif model.application_map[cl] == PROTO_DICT['gnutella']:
                            gnutella_tp += 1
                        elif model.application_map[cl] == PROTO_DICT['edonkey']:
                            edonkey_tp += 1
                        elif model.application_map[cl] == PROTO_DICT['bittorrent']:
                            bittorrent_tp += 1
                        take_break = True
                        break
                    else:
                         false_positive += 1
                         if model.application_map[cl] == PROTO_DICT['http']:
                             http_fp += 1
                         elif model.application_map[cl] == PROTO_DICT['gnutella']:
                             gnutella_fp += 1
                         elif model.application_map[cl] == PROTO_DICT['edonkey']:
                             edonkey_fp += 1
                         elif model.application_map[cl] == PROTO_DICT['bittorrent']:
                             bittorrent_fp += 1
                         take_break = True
                         break
            if take_break:
                break
            # if result == model.clusters_with_high_purity and result not in model.application_map.keys():
            #     false_positive += 1
            #     if model.application == PROTO_DICT['http']:
            #         http_fp += 1
            #     elif model.application == PROTO_DICT['gnutella']:
            #         gnutella_fp += 1
            #     elif model.application == PROTO_DICT['edonkey']:
            #         edonkey_fp += 1
            #     elif model.application == PROTO_DICT['bittorrent']:
            #         bittorrent_fp += 1
            #     break
    print(true_positive)
    print(float(true_positive)/total_flows)
    return (float(true_positive)/total_flows, float(http_tp)/http_test.shape[0], 
            float(gnutella_tp)/gnutella_test.shape[0], float(edonkey_tp)/edonkey_test.shape[0], 
            float(bittorrent_tp)/bittorrent_test.shape[0], true_positive, http_tp, gnutella_tp, 
            edonkey_tp, bittorrent_tp, http_test.shape[0], gnutella_test.shape[0], 
            edonkey_test.shape[0], bittorrent_test.shape[0], false_positive, http_fp, gnutella_fp, edonkey_fp, bittorrent_fp)

def main(use_minimum_data = False):
    global classification_model_list
    results_map = {}
    number_of_iterations = 5
    first = True
    with open('results_latest_temp2.txt', 'w', 5) as output_file:
        #for percent in np.linspace(0.0, 0.99, 10):
        for percent in [0.90, ]:
            http_accuracy = 0.0
            gnutella_accuracy = 0.0
            edonkey_accuracy = 0.0
            bittorrent_accuracy = 0.0
            http_tp = 0
            gnutella_tp = 0
            edonkey_tp = 0
            bittorrent_tp = 0
            total_accuracy = 0.0
            total_tp = 0
            total_false_positive = 0
            total_false_negative = 0
            http_false_positive = 0
            http_false_negative = 0
            gnutella_false_positive = 0
            gnutella_false_negative = 0
            edonkey_false_positive = 0
            edonkey_false_negative = 0
            bittorrent_false_positive = 0
            bittorrent_false_negative = 0
            http_test_shape = 0
            gnutella_test_shape = 0
            edonkey_test_shape = 0
            bittorrent_test_shape = 0

            false_positive = 0
            http_fp = 0
            gnutella_fp = 0
            edonkey_fp = 0
            bittorrent_fp = 0

            if first:
                first = False
                continue
            total_value = 0.0
            for i in range(number_of_iterations):
                print('Percent: '+str(percent))
                classification_model_list = []
                my_data = np.genfromtxt(OUTPUT_FILE, delimiter=",")
                if use_minimum_data:
                    my_data = get_data(my_data, 100)
                nrows, ncols = my_data.shape
                total_idx = np.arange(nrows)
                train_idx = np.random.randint(my_data.shape[0], size=int(nrows*percent))
                print(train_idx)
                #test_idx = np.random.randint(my_data.shape[0], size=int(nrows*(1.0 - percent)))
                test_idx = np.delete(total_idx, train_idx, 0)
                print(test_idx)
                k_means(my_data[train_idx, :])
                my_test_data = my_data[test_idx, :]
                print('Test Data Shape: '+str(my_test_data.shape))
                temp_total_accuracy, temp_http_accuracy, temp_gnutella_accuracy, temp_edonkey_accuracy, temp_bittorrent_accuracy, temp_total_tp, temp_http_tp, temp_gnutella_tp, temp_edonkey_tp, temp_bittorrent_tp, temp_http_test_shape, temp_gnutella_test_shape, temp_edonkey_test_shape, temp_bittorrent_test_shape, temp_false_positive, temp_http_fp, temp_gnutella_fp, temp_edonkey_fp, temp_bittorrent_fp = test_classification_model(my_test_data, percent, output_file)
                total_accuracy += temp_total_accuracy
                http_accuracy += temp_http_accuracy
                gnutella_accuracy += temp_gnutella_accuracy
                edonkey_accuracy += temp_edonkey_accuracy
                bittorrent_accuracy += temp_bittorrent_accuracy
                total_tp += temp_total_tp
                http_tp += temp_http_tp
                gnutella_tp += temp_gnutella_tp
                edonkey_tp += temp_edonkey_tp
                bittorrent_tp += temp_bittorrent_tp
                http_test_shape += temp_http_test_shape
                gnutella_test_shape += temp_gnutella_test_shape
                edonkey_test_shape += temp_edonkey_test_shape
                bittorrent_test_shape += temp_bittorrent_test_shape

                false_positive += temp_false_positive
                http_fp += temp_http_fp
                gnutella_fp += temp_gnutella_fp
                edonkey_fp += temp_edonkey_fp
                bittorrent_fp += temp_bittorrent_fp

            results_map[percent] = total_accuracy/number_of_iterations
            total_accuracy = total_accuracy/number_of_iterations
            http_accuracy = http_accuracy/number_of_iterations
            bittorrent_accuracy = bittorrent_accuracy/number_of_iterations
            gnutella_accuracy = gnutella_accuracy/number_of_iterations
            edonkey_accuracy = edonkey_accuracy/number_of_iterations
            total_tp = total_tp/number_of_iterations
            
            http_tp = http_tp/number_of_iterations
            http_test_shape = http_test_shape/number_of_iterations
            http_false_positive = http_fp/number_of_iterations
            http_false_negative = http_test_shape - http_tp

            gnutella_tp = gnutella_tp/number_of_iterations
            gnutella_test_shape = gnutella_test_shape/number_of_iterations
            gnutella_false_positive = gnutella_fp/number_of_iterations
            gnutella_false_negative = gnutella_test_shape - gnutella_tp

            edonkey_tp = edonkey_tp/number_of_iterations
            edonkey_test_shape = edonkey_test_shape/number_of_iterations
            edonkey_false_positive = edonkey_fp/number_of_iterations
            edonkey_false_negative = edonkey_test_shape - edonkey_tp

            bittorrent_tp = bittorrent_tp/number_of_iterations
            bittorrent_test_shape = bittorrent_test_shape/number_of_iterations
            bittorrent_false_positive = bittorrent_fp/number_of_iterations
            bittorrent_false_negative = bittorrent_test_shape - bittorrent_tp

            output_file.write("Total Accuracy :"+str(total_accuracy) + '\n')
            output_file.write("Http Accuracy :"+str(http_accuracy) + '\n')
            output_file.write("Edonkey Accuracy :"+str(edonkey_accuracy) + '\n')
            output_file.write("Gnutella Accuracy :"+str(gnutella_accuracy) + '\n')
            output_file.write("Bittorrent Accuracy :"+str(bittorrent_accuracy) + '\n')

            output_file.write("Http True Positive :"+str(http_tp) + '\n')
            output_file.write("Http False Positive :"+str(http_false_positive) + '\n')
            output_file.write("Http False Negative :"+str(http_false_negative) + '\n')

            output_file.write("Gnutella True Positive :"+str(gnutella_tp) + '\n')
            output_file.write("Gnutella False Positive :"+str(gnutella_false_positive) + '\n')
            output_file.write("Gnutella False Negative :"+str(gnutella_false_negative) + '\n')

            output_file.write("Edonkey True Positive :"+str(edonkey_tp) + '\n')
            output_file.write("Edonkey False Positive :"+str(edonkey_false_positive) + '\n')
            output_file.write("Edonkey False Negative :"+str(edonkey_false_negative) + '\n')

            output_file.write("Bittorrent True Positive :"+str(bittorrent_tp) + '\n')
            output_file.write("Bittorrent False Positive :"+str(bittorrent_false_positive) + '\n')
            output_file.write("BIttorrent False Negative :"+str(bittorrent_false_negative) + '\n')

            output_file.write('='*15+' '+str(percent) + ' '+'='*15+'\n')
            print('Final Accuracy is: '+str(total_value/number_of_iterations))

    print(results_map)
    plt.plot(results_map.keys(), results_map.values(), 'ro')
    plt.xlabel('Percent of % training set')
    plt.ylabel('Accuracy TP/Total')
    plt.title('Percent of Training Set v/s Accuracy')
    plt.show()

if __name__=='__main__':
    main(True)