import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class FriendsRecModel(object):
    def __init__(self):
        self.entire = None
        self.sub = None
        self.euclid = None
        self.cos_simi = None
        self.eval_val = None
        self.eval_sort = None
    
    def id_check(self):
        for i in range(len(self.entire)):
            if (self.entire[i,4] == self.sub[0,4]):
                temp = np.delete(self.entire,i,0)
        self.entire = temp
        
    def calc_euclid(self):
        for i in range(len(self.entire)):
            self.euclid[i,0] = np.linalg.norm(self.entire[i,:4]-self.sub[0,:4])
            self.euclid[i,1] = self.entire[i,4]
        
    def calc_cos_simi(self):
        for i in range(len(self.entire)):
            self.cos_simi[i,0] = cosine_similarity(self.entire[i:i+1,:4],self.sub[0:1,:4]) 
            self.cos_simi[i,1] = self.entire[i,4]
            
    def min_max_scalar(self, X):
        mm = MinMaxScaler()
        mm.fit(X[:,0].reshape(len(X[:,0]),1))
        X[:,0:1] = mm.transform(X[:,0].reshape(len(X[:,0]),1))
        return X
    
    def calc_eval(self):
        self.euclid = self.min_max_scalar(self.euclid)
        self.cos_simi = self.min_max_scalar(self.cos_simi)
        self.eval_val = self.euclid
        
        self.eval_val[:,0] = self.cos_simi[:,0]-self.euclid[:,0]
        self.eval_val[:,1] = self.cos_simi[:,1]
    
    def sort_eval(self):
        self.eval_sort = self.eval_val[self.eval_val[:,0].argsort(),:][::-1]
        
    def forward(self, entire, sub):
        self.entire = np.array(entire).reshape(-1,5)
        self.sub = np.array(sub).reshape(-1,5)
        self.id_check()
        
        self.euclid = np.zeros((len(self.entire),2))
        self.cos_simi = np.zeros((len(self.entire),2))
        
        self.calc_euclid()
        self.calc_cos_simi()
        self.calc_eval()
        self.sort_eval()
        return self.eval_sort