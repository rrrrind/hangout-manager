import numpy as np
from static_model import RidgeRegModel as rrm
from dynamic_model import RidgeClfModel as rcm
from friends_model import FriendsRecModel as frm

class RecommendHangout(object):
    def __init__(self, static_q_list, dynamic_q_list):
        self.static_q_list = np.array(static_q_list).reshape(-1,8)
        self.dynamic_q_list = np.array(dynamic_q_list).reshape(-1,3)
        self.static_model = rrm()
        self.dynamic_model = rcm()

    def recommend_genre(self, result):
        if result == 0 :
            return "アウトドア"
        elif result == 1 :
            return "エンタメ"
        elif result == 2 :
            return "スポーツ"
        elif result == 3 :
            return "観光"
        elif result == 4 :
            return "レジャーランド等"
        elif result == 5 :
            return "友達との飲み会"
        elif result == 6 :
            return "外での買い物"
        elif result == 7 :
            return "ショー・展示・催し"


    def recommend_hangouts(self, result):
        if result == 0 :
            return ["BBQ","日帰りキャンプ","釣り","パラグライダー","トレッキング"]
        elif result == 1 :
            return ["カラオケ","ボウリング場","ゲームセンター","ビリヤード","ダーツ"]
        elif result == 2 :
            return ["ボルダリング","ヨガ","フィットネスジム","バッティングセンター","卓球"]
        elif result == 3 :
            return ["観光名所巡り","日帰り温泉","食べ歩き","バスツアー","アニメなどの聖地巡礼"]
        elif result == 4 :
            return ["遊園地","動物園","プール","水族館","海水浴場"]
        elif result == 5 :
            return ["オンライン飲み会","飲み歩き","宅飲み","利き酒","合コン"]
        elif result == 6 :
            return ["アウトレット","古着屋巡り","ショッピングモール","商店街","フリーマーケット"]
        elif result == 7 :
            return ["映画館","ミュージカル","プラネタリウム","演劇・舞台","博物館・美術館"]


    def forward(self): 
        s_output = self.static_model.forward(self.static_q_list)
        result = self.dynamic_model.forward(s_output, self.dynamic_q_list[:,:2])
        return self.recommend_genre(result), self.recommend_hangouts(result)
    

class RecommendFriends(object):
    def __init__(self, entire_q_list, sub_q_list):
        self.entire_q_list = entire_q_list
        self.sub_q_list = sub_q_list
        self.friends_model = frm()
        
    def forward(self):
        return self.friends_model.forward(self.entire_q_list, self.sub_q_list)