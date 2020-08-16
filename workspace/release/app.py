from flask import *
import requests
import json
import urllib3

from main import RecommendHangout
from main import RecommendFriends

app = Flask(__name__)
http = urllib3.PoolManager()

@app.route("/hangouts/<hang_out_id>")
def get_hangout(hang_out_id):
    url = 'http://web:3000/hang_outs/' + hang_out_id + '/json'
    res = http.request('GET',url)
    return res.data.decode('utf-8')

@app.route("/current/<user_id>")
def get_current_user(user_id):
    url = 'http://web:3000/to_current/' + user_id + '/json'
    res = http.request('GET',url)
    data = json.loads(res.data.decode('utf-8'))
    current_list = [data['p_one'], data['p_two'], data['p_three'], data['p_four'], data['user_id']]
    print (data)
    print (current_list)
    return res.data.decode('utf-8')

@app.route("/questions/<question_id>")
def get_question(question_id):
    url = 'http://web:3000/questions/' + question_id + '/json'
    res = http.request('GET',url)
    data = json.loads(res.data.decode('utf-8'))
    q1 = data['question']['q_one']
    q2 = data['question']['q_two']
    q3 = data['question']['q_three']
    d1 = data['user_detail']['d_one']
    d2 = data['user_detail']['d_two']
    d3 = data['user_detail']['d_three']
    d4 = data['user_detail']['d_four']
    d5 = data['user_detail']['d_five']
    d6 = data['user_detail']['d_six']
    d7 = data['user_detail']['d_seven']
    d8 = data['user_detail']['d_eight']
    print (data)
    rh = RecommendHangout([d1,d2,d3,d4,d5,d6,d7,d8],[q1,q2,q3])
    genre, hangouts = rh.forward()
    return_rec = dict(a1=hangouts[0],
                      a2=hangouts[1],
                      a3=hangouts[2],
                      a4=hangouts[3],
                      a5=hangouts[4],
                      ge=genre)

    return return_rec

@app.route("/user_details/<user_detail_id>")
def get_user_detail(user_detail_id):
    url = 'http://web:3000/user_details/' + user_detail_id + '/json'
    res = http.request('GET',url)
    user_detail_json = json.loads(res.data.decode('utf-8'))
    d1 = user_detail_json['d_one']
    d2 = user_detail_json['d_two']
    d3 = user_detail_json['d_three']
    d4 = user_detail_json['d_four']
    d5 = user_detail_json['d_five']
    d6 = user_detail_json['d_six']
    d7 = user_detail_json['d_seven']
    d8 = user_detail_json['d_eight']
    print ('d1', d1, 'd2', d2, 'd3', d3, 'd4', d4, 'd5', d5, 'd6', d6, 'd7', d7, 'd8', d8)
    return res.data.decode('utf-8')

@app.route("/user_personals/<user_personal_id>")
def get_user_personal(user_personal_id):
    url = 'http://web:3000/user_personals/' + user_personal_id + '/json'
    res = http.request('GET',url)
    user_personal_json = json.loads(res.data.decode('utf-8'))
    p1 = user_personal_json['p_one']
    p2 = user_personal_json['p_two']
    p3 = user_personal_json['p_three']
    p4 = user_personal_json['p_four']
    print ('p1', p1, 'p2', p2, 'p3', p3, 'p4', p4)
    return res.data.decode('utf-8')

@app.route("/users")
def get_users():
    url = 'http://web:3000/user_all/json'
    res = http.request('GET',url)
    users_json = json.loads(res.data.decode('utf-8'))
    user_list = []
    user_personal_list = []
    for i in range(len(users_json['user_personals'])):
        user_personal_list += [users_json['user_personals'][i]['p_one']]
        user_personal_list += [users_json['user_personals'][i]['p_two']]
        user_personal_list += [users_json['user_personals'][i]['p_three']]
        user_personal_list += [users_json['user_personals'][i]['p_four']]
        user_personal_list += [users_json['user_personals'][i]['user_id']]
        user_list += [user_personal_list]
        user_personal_list = []

    print (user_list) # user_list->Userの情報[p_one, p_two, p_three, p_four, user_id]
    return res.data.decode('utf-8')

@app.route("/friend/<current_user_id>")
def get_friend(current_user_id):
    url_all = 'http://web:3000/user_all/json'
    res_all = http.request('GET',url_all)
    users_json = json.loads(res_all.data.decode('utf-8'))
    user_list = []
    user_personal_list = []
    for i in range(len(users_json['user_personals'])):
        user_personal_list += [users_json['user_personals'][i]['p_one']]
        user_personal_list += [users_json['user_personals'][i]['p_two']]
        user_personal_list += [users_json['user_personals'][i]['p_three']]
        user_personal_list += [users_json['user_personals'][i]['p_four']]
        user_personal_list += [users_json['user_personals'][i]['user_id']]
        user_list += [user_personal_list]
        user_personal_list = []

    url_current = 'http://web:3000/to_current/' + current_user_id + '/json'
    res_current = http.request('GET',url_current)
    current_json = json.loads(res_current.data.decode('utf-8'))
    current_list = [current_json['p_one'], current_json['p_two'], current_json['p_three'], current_json['p_four'], current_json['user_id']]
    
    rf = RecommendFriends(user_list,current_list)
    rf_rank = rf.forward()   
    return_rank = dict(ID0=int(rf_rank[0,1]),
                       ID1=int(rf_rank[1,1]),
                       ID2=int(rf_rank[2,1]),
                       ID3=int(rf_rank[3,1]),
                       ID4=int(rf_rank[4,1]),
                       ID5=int(rf_rank[5,1]),
                       ID6=int(rf_rank[6,1]),
                       ID7=int(rf_rank[7,1]),
                       ID8=int(rf_rank[8,1]),
                       ID9=int(rf_rank[9,1]))    
    return return_rank

## おまじない
if __name__ == "__main__":
    app.run(debug=True)