import numpy as np
import sys
from numpy.fft import fft
import matplotlib as plot
from flask import Flask, render_template, request, url_for, jsonify
from flask_cors import CORS, cross_origin
import json

num_top = 10
quake_start = np.load('static/quake_start.npy')
data = np.load('static/topic_weights_shear.npy')
totalnum = len(quake_start) - 1

app = Flask(__name__)


def jsonifydata(data):
    data_new = []
    for i, eq in enumerate(data):
        for j, v in enumerate(eq):
            data_new.append({"i": i, "j": j, "value": v})
    return jsonify(data_new)


def jsonifyresultdata(topmatchlist, length):
    #print("length", length)

    def jsonifyactualdata(data):
        data_new = []
        for i, d in enumerate(data):
            for j, v in enumerate(d):
                data_new.append({"i": i, "j": j, "value": v})
        return data_new

    data_result = []
    for matches in topmatchlist:
        result_value, result_index, result_eqnum = matches[0], matches[1], matches[2]

        eq_result = data[quake_start[result_eqnum - 1]:quake_start[result_eqnum], :].T
        #print(result_value, result_index, type(result_eqnum), eq_result.shape, length)
        '''
        if (result_index + length ) <= eq_result.shape[1]:
            eq_seg = [i[result_index:result_index+length] for i in eq_result]
            #print("don't exceed",len(eq_seg),len(eq_seg[0]))
        else:
            eq_seg=[ np.concatenate((i[result_index:len(i)+1], i[0: length-(len(i)-result_index)])) for i in eq_result]
            #print("exceed",len(eq_seg),len(eq_seg[0]))
        '''
        pass_data = jsonifyactualdata(eq_result)
        data_result.append({"eqnum": result_eqnum, "data": pass_data, "value": result_value, "index": int(result_index), "length": int(length)})
    return jsonify(data_result)


def maxshiftinnerproduct(v1, v2):
    cs = np.sum(np.cumsum(v2 ** 2, axis=1), axis=0)
    # print(cs)

    def range_sum(i, j):
        if i <= 0:
            lo = 0
        elif i >= len(cs):
            lo = cs[-1]
        else:
            lo = cs[i - 1]
        if j <= 0:
            hi = 0
        elif j >= len(cs):
            hi = cs[-1]
        else:
            hi = cs[j - 1]
        return hi - lo

    def wrapping_range_sum(i, l):
        # print(i, l)
        if i + l > len(cs):
            return range_sum(i, len(cs) - 1) + range_sum(0, len(cs) - i - l)
        else:
            return range_sum(i, i + l)

    # print(v1.shape)
    # print(v2.shape)
    # v1 is the smaller time interval
    v1 = np.array(v1)
    l = v1.shape[1]
    v2 = np.array(v2)
    # print(v1.shape, v2.shape)
    diff = v2.shape[1] - v1.shape[1]
    if diff > 0:
        s = np.zeros((v1.shape[0], diff))
        # print(s.shape)
        v1 = np.concatenate((v1, s), axis=1)
    else:
        s = np.zeros((v1.shape[0], -diff))
        v2 = np.concatenate((v2, s), axis=1)
    Ts = 1
    Fs = 1.0 / Ts
    n = len(v1)
    f = np.arange(n, dtype=np.float64) / n * Fs
    # This fft could be speed up bying pre store the fft results or wraping the lenth to power of 2
    xf1 = fft(v1[::, ::-1])
    xf2 = fft(v2)
    product = xf1 * xf2
    xt = np.fft.ifft(product)
    xt = xt.real
    xt = np.sum(xt, axis=0)
    a = np.array([wrapping_range_sum(i, l) for i in range(v2.shape[1])])
    b = xt
    c = np.sum(v1 ** 2)
    # print(c)
    d = a - 2 * b + c
    # print(xt)
    return abs(np.min(d)), np.argmin(d)

# new method using numpy.correlate


def maxshiftinnerproductv2(v1, v2):
    # if v1 is short than v2, flag =0, otherwise flag =1
    flag = 0
    v1 = np.array(v1)
    v2 = np.array(v2)
    if (v1.shape[1] > v2.shape[1]):
        temp = v1
        v1 = v2
        v2 = temp
        flag = 1
    cs = np.sum(np.cumsum(v2 ** 2, axis=1), axis=0)
    l = v1.shape[1]
    # print(cs)

    def range_sum(i, j):
        if i <= 0:
            lo = 0
        elif i >= len(cs):
            lo = cs[-1]
        else:
            lo = cs[i - 1]
        if j <= 0:
            hi = 0
        elif j >= len(cs):
            hi = cs[-1]
        else:
            hi = cs[j - 1]
        return hi - lo

    def wrapping_range_sum(i, l):
        # print(i, l)
        if i + l > len(cs):
            # return range_sum(i, len(cs) - 1) + range_sum(0, len(cs) - i - l)
            return range_sum(i, len(cs) - 1)
        else:
            return range_sum(i, i + l)

    a = np.array([wrapping_range_sum(i, l) for i in range(v2.shape[1] - v1.shape[1] + 1)])
    b = np.sum([np.correlate(v1[i], v2[i])[::-1] for i in range(v1.shape[0])], axis=0)
    c = np.sum(v1 ** 2)
    # print("c:",c)
    # print("a:",a)
    # print("b:",b)
    d = a - 2 * b + c

    return abs(np.min(d)), np.argmin(d)


# this will generate a random sequence, and the sum of the column is one


def randompercentagesequence(l, t):
    input = np.random.dirichlet(np.ones(t), size=1).T
    for i in range(l - 1):
        a = np.random.dirichlet(np.ones(t), size=1)
        input = np.concatenate((input, a.T), axis=1)
    return input


def NN_Search(input):
    match_list = []
    for i in range(len(quake_start) - 1):
        # for i in range(1):
        eq = data[quake_start[i]:quake_start[i + 1], :].T
        if (len(input[0]) <= len(eq[0])):
            maxvalue, index = maxshiftinnerproductv2(input, eq)
            match_list.append([maxvalue, index, i + 1])

    match_list = sorted(match_list)

    return match_list[0:num_top]


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('query.html', totalnum=totalnum)


@app.route('/data', methods=["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_data():
    eqnum = json.loads(request.get_data())
    eqnum = int(eqnum)
    eq = data[quake_start[eqnum - 1]:quake_start[eqnum], :].T
    return jsonifydata(eq)


@app.route('/data_query', methods=["GET", "POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_data_query():
    dict_index = json.loads(request.get_data())
    start, end, eqnum = int(dict_index['start']), int(dict_index['end']), int(dict_index['eqnum'])
    length = end - start + 1
    eq = data[quake_start[eqnum - 1]:quake_start[eqnum], :].T
    input = [i[start:end + 1] for i in eq]
    topmatchlist = NN_Search(input)
    return jsonifyresultdata(topmatchlist, length)


'''

@app.route('/user_study')
def user_study():
    images = get_images()
    username = request.args.get('username')
    page = request.args.get('page')
    choice = request.args.get('choice')

    if page == None:
        page = 0
    else:
        page = int(page)
        if page == pairoftest * 2:
            save_response(sessionid, username, 1, choice, images[page - 1])
            return render_template('end.html')
    if choice != None:
        save_response(sessionid, username, 1, choice, images[page - 1])
    urls = [url_for('static', filename=images_path + img) for img in images[page]]
    return render_template('user_study.html', username=username, images=urls, page=page)
'''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
