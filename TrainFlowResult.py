#!/usr/bin/env python

class TrainFlowResult(object):
    def __init__(self):
        self.group = None
        self.probability_for_http = None
        self.probability_for_gnutella = None
        self.probability_for_edonkey = None
        self.probability_for_bittorrent = None
        self.final_probability = None
        self.application = None

    def calculate_final_probability(self):
        #self.final_probabilty = 1.0 - ((1.0 - self.probability_for_app1)*(1.0 - self.probability_for_app2)*(1.0 - self.probability_for_app3))
        self.final_probability = max(self.probability_for_http, self.probability_for_gnutella, self.probability_for_edonkey, self.probability_for_bittorrent)
        if self.final_probability == self.probability_for_http:
            self.application = 0.0
        elif self.final_probability == self.probability_for_gnutella:
            self.application = 1.0
        elif self.final_probability == self.probability_for_edonkey:
            self.application = 2.0
        elif self.final_probability == self.probability_for_bittorrent:
            self.application = 3.0