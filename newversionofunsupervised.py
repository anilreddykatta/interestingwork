__author__ = 'KattaAnil'


import numpy as np
from sklearn.cluster import KMeans
from TrainFlowResult import TrainFlowResult
from TestFlowResult import TesFlowResult
from ProbabilityDistribution import ProbabilityDistribution
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

PROTO_DICT_MAP = {0: 'http', 1: 'gnutella', 2: "edonkey", 3: "bittorrent"}

PROTO_DICT = {'http': 0, 'gnutella': 1, "edonkey": 2, "bittorrent": 3}

DEBUG = False

REMOVE_CLUSTER_WITH_LESS_FLOWS = False

def get_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")


def return_train_test_labels(input_data, number_of_each_flows, percent=0.9):
    input_data = np.copy(input_data)
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

def attribute_group_for_testing(data, attribute_group, flow_number):
    if attribute_group == 1:
        return data[flow_number, :4]
    elif attribute_group == 2:
        return data[flow_number, 4:8]
    elif attribute_group == 3:
        return data[flow_number, 8:]

def get_indices_for_pred(pred, cluter_number):
    return np.where(pred == cluter_number)


def generate_stats_for_cluster(data, cluster_number, attribute_group):
    result_map = {0: 0, 1: 0, 2: 0, 3: 0}
    #print(data.tolist())
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
        if i == PROTO_DICT['http']:
             st_http = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['gnutella']:
            st_gnutella = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['edonkey']:
            st_edonkey = float(result_map[i])/sum(result_map.values())
        elif i == PROTO_DICT['bittorrent']:
            st_bittorrent = float(result_map[i])/sum(result_map.values())
    if DEBUG:
        print(st.format('Cluster'+str(cluster_number), st_http, st_gnutella, st_edonkey, st_bittorrent))
    train_flow_result = TrainFlowResult()
    train_flow_result.group = attribute_group
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
        if REMOVE_CLUSTER_WITH_LESS_FLOWS and indices.size < 10:
            continue
        refined_data = np.copy(data)
        refined_data = combine_labels_data(refined_data, train_labels)
        refined_data = refined_data[indices, -1]
        train_result_map[i] = generate_stats_for_cluster(refined_data, i, attribute_group)

    #Testing Begins here
    return (k_means_cls, train_result_map)

def testing(test_data, test_labels, k_means_cls, train_result_map, attribute_group, csv_file, group_number):
    pred = k_means_cls.predict(test_data)
    index = 0
    test_flow_result = None
    true_positive = 0
    results_map = {}
    for i in pred.tolist():
        test_flow_result = TesFlowResult()
        test_flow_result.actual_protocol = PROTO_DICT_MAP[int(test_labels[index, :][0])]
        test_flow_result.classification_result = PROTO_DICT_MAP[int(train_result_map[int(i)].application)]
        test_flow_result.probability = train_result_map[int(i)].final_probability
        test_flow_result.testing_flow = index
        test_flow_result.true_positive = (PROTO_DICT_MAP[int(test_labels[index, :][0])] == PROTO_DICT_MAP[int(train_result_map[int(i)].application)])
        if test_flow_result.true_positive:
            true_positive += 1
        test_flow_result.write_to_csv(csv_file)
        index += 1
        if test_flow_result.classification_result:
            #results_map[test_flow_result.probability] = test_flow_result.classification_result
            results_map[test_flow_result.probability] = 2.0
        else:
            results_map[test_flow_result.probability] = 1.0
    print('Final Accuracy: '+str(float(true_positive)/index))
    return results_map

def main(number_of_clusters=20, number_of_times=10):
    input_data = get_numpy_array_from_file("realinput.csv")
    train, train_labels, test, test_labels = return_train_test_labels(input_data, None)
    fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
    k_means_map = {}
    attribute_train_result_map = {}
    for i in range(1, 4):
        with open('resultsfinalgroup'+str(i)+'.csv', 'w', 20) as output_file:
            csv_file = csv.DictWriter(output_file, delimiter=",", fieldnames=fieldnames)
            csv_file.writeheader()
            if DEBUG:
                print('{0:15}{1:10}{2:10}{3:10}{4:10}'.format('Cluster', 'Http', 'Gnutella', 'Edonkey', 'Bittorrent'))
            after_attribute_set = get_attribute_group(train, i)
            k_means_cls, train_result_map = k_means(after_attribute_set, train_labels, number_of_clusters, number_of_times, i)
            k_means_map[i] = k_means_cls
            attribute_train_result_map[i] = train_result_map    
    after_attribute_test_set = get_attribute_group(test, i)
    #probability_result_map = testing(after_attribute_test_set, test_labels, k_means_cls, train_result_map, i, csv_file)
    test_testing_set(test, attribute_train_result_map, k_means_map, test_labels)
    print('='*15+str('Atrribute Group '+str(i))+'='*15)


def test_testing_set(test_data, attribute_results_map, k_means_map, test_labels):
    results_map = {}
    plot_map = {}
    probs = []
    probability_distribution = None
    true_positive = 0
    with open('experiment.log', 'w', 20) as output_file:
        for i in range(test_data.shape[0]):
            probability_distribution = ProbabilityDistribution()
            for j in range(1, 4):
                test_set = attribute_group_for_testing(test_data, j, i)
                predicted_value = k_means_map[j].predict(test_set)
                if REMOVE_CLUSTER_WITH_LESS_FLOWS and (int(predicted_value) not in attribute_results_map[j].keys()):
                    continue
                train_result = attribute_results_map[j][int(predicted_value)]
                if j == 1:
                    probability_distribution.http_g1 = train_result.probability_for_http
                    probability_distribution.gnutella_g1 = train_result.probability_for_gnutella
                    probability_distribution.edonkey_g1 = train_result.probability_for_edonkey
                    probability_distribution.bittorrent_g1 = train_result.probability_for_bittorrent
                elif j == 2:
                    probability_distribution.http_g2 = train_result.probability_for_http
                    probability_distribution.gnutella_g2 = train_result.probability_for_gnutella
                    probability_distribution.edonkey_g2 = train_result.probability_for_edonkey
                    probability_distribution.bittorrent_g2 = train_result.probability_for_bittorrent
                elif j == 3:
                    probability_distribution.http_g3 = train_result.probability_for_http
                    probability_distribution.gnutella_g3 = train_result.probability_for_gnutella
                    probability_distribution.edonkey_g3 = train_result.probability_for_edonkey
                    probability_distribution.bittorrent_g3 = train_result.probability_for_bittorrent    
            prob, app = probability_distribution.get_final_probability()
            output_file.write(str(probability_distribution))
            output_file.write('--'*30+'\n')
            if app == PROTO_DICT_MAP[int(test_labels[i, -1])]:
                true_positive += 1
            else:
                prob = float(int(prob*100))/100
                probs.append(prob)
                pass
            if prob in plot_map.keys():
                plot_map[prob] = plot_map[prob] + 1
            else:
                plot_map[prob] = 1
            results_map[i] = probability_distribution
        print('Final Accuracy: '+str(float(true_positive)/test_data.shape[0]))
        x = plot_map.keys()
        y = plot_map.values()
        #plt.plot(x, norm.pdf(y), 'r-', lw=5, alpha=0.6, label='norm pdf')
        #plt.plot(probs, norm.pdf(probs), 'r-', lw=5, alpha=0.6, label='norm pdf')
        #plt.plot(plot_map.keys(), plot_map.values())
    
        n, bins, patches = plt.hist(probs, 50, normed=1, facecolor='green', alpha=0.75)
        import matplotlib.mlab as mlab
        #y = mlab.normpdf(bins, mu, sigma)
        #l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.show()

def func(x, a, b, c, d):
    #return a * np.sin(b*x + c) + d
    return (np.e**(-x*x/2))/np.sqrt(2*np.pi)
if __name__ == '__main__':
    main(25, 10)