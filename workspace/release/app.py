from flask import *
import requests
import json
import urllib3

from main import RecommendHangout

app = Flask(__name__)
http = urllib3.PoolManager()

@app.route("/hangouts/<hang_out_id>")
def get_hangout(hang_out_id):
    url = 'http://web:3000/hang_outs/' + hang_out_id + '/json'
    res = http.request('GET',url)
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
    p5 = user_personal_json['p_five']
    p6 = user_personal_json['p_six']
    print ('p1', p1, 'p2', p2, 'p3', p3, 'p4', p4, 'p5', p5, 'p6', p6)
    return res.data.decode('utf-8')

## おまじない
if __name__ == "__main__":
    app.run(debug=True)

