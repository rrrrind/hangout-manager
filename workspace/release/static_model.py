import pickle
import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegModel(object):
    def __init__(self):
        with open('./trainned_model/static_model.pickle', 'rb') as f_static:
            self.model = pickle.load(f_static)

    def forward(self, questions):
        return self.model.predict(questions)
