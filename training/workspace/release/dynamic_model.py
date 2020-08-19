import pickle
import numpy as np
from sklearn.linear_model import RidgeClassifier

class RidgeClfModel(object):
    def __init__(self):
        with open('./trainned_model/dynamic_model.pickle', 'rb') as f_dynamic:
            self.model = pickle.load(f_dynamic)

    def forward(self, stat_output:list, questions:list):
        inputs = np.hstack((stat_output, questions))
        return self.model.predict(inputs)
