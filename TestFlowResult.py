#!/usr/bin/env python


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

    def write_to_csv(self, csv_file):
        fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
        csv_file.writerow({'Testing Flow':self.testing_flow, 'Actual Protocol':self.actual_protocol , 
                           'Classification Result': self.classification_result, 'Probability':self.probability, 
                           'True Positive':self.true_positive})

    def calculate_avg_population_fraction(self):
        self.avg_population_fraction = sum(self.group_flow_one_population + self.group_flow_two_population + 
                                           self.group_pkt_size_one_population + self.group_pkt_size_two_population) / 4.0

    """def calculate_simple_probability(self):
        bit_prob = 1.0 - (1.0 - self.group_pkt_size_one_population)*(1.0 - self.bittorrent_g2)*(1.0 - self.bittorrent_g3)
        http_prob = 1.0 - (1.0 - self.http_g1)*(1.0 - self.http_g2)*(1.0 - self.http_g3)
        gnu_prob = 1.0 - (1.0 - self.gnutella_g1)*(1.0 - self.gnutella_g2)*(1.0 - self.gnutella_g3)
        edo_prob = 1.0 - (1.0 - self.edonkey_g1)*(1.0 - self.edonkey_g2)*(1.0 - self.edonkey_g3)"""