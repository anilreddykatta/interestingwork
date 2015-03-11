__author__ = 'KattaAnil'


class ClusterClassificationModel(object):
    def __init__(self, iteration, model, attribute):
        self.iteration = iteration
        self.model = model
        self.attribute = attribute
        self.application_map = {}
        #self.purity = purity

    def __str__(self):
        return 'Iteration: '+str(self.iteration)+' application: '+str(self.application_map)+' attribute: '+str(self.attribute)

    def __repr__(self):
        return 'Iteration: '+str(self.iteration)+' application: '+str(self.application_map)+' attribute: '+str(self.attribute)

    def predict(self, data_point):
        pass
