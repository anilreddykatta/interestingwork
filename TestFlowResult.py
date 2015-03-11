#!/usr/bin/env python


class TesFlowResult(object):

    def __init__(self):
        self.testing_flow = None
        self.actual_protocol = None
        self.classification_result = None
        self.probability = None
        self.true_positive = None

    def write_to_csv(self, csv_file):
        fieldnames = ['Testing Flow', 'Actual Protocol', 'Classification Result', 'Probability', 'True Positive']
        csv_file.writerow({'Testing Flow':self.testing_flow, 'Actual Protocol':self.actual_protocol , 
                           'Classification Result': self.classification_result, 'Probability':self.probability, 
                           'True Positive':self.true_positive})