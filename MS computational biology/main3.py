#panda.read_csv##ln31#get a txt file that looks like excel or csv into panda dataframe
#panda dataframe.iloc##ln34# slices dataframe based on positions
#enumerate## create 2 1d array ones with indexes and the other with element of an array
#PCA## create 
#
#
#
#
#
#
#
#
#
#
#
import sys
import numpy as np
from string import *
import collections
from numpy.random import random
import dash
import dash_core_components as dcc
import dash_html_components as html
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.covariance import GraphLassoCV
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib.colors import SymLogNorm

import networkx as nx


def main(argv):
    formatted = read_data_from_file(argv[1])  ## this store the data in my class: DataPoint
    # for data_point in formatted:
    #     print data_point.features.featuresArray
    #     print data_point.relevance
    #     print data_point.q_id
    #     print data_point.doc_id

    x = pd.read_csv(argv[1], sep=',', header=None) 
    features = x.iloc[:, 2:66].astype(np.float64)

    # Basic_count(x,formatted)
    # Rough_plot(formatted)
    # plotAllD(formatted)
    # principle_component_analysis(formatted, features)
    # logisticRe(formatted, features)
    # matrix_analysis(argv)


# This is just to count something
def Basic_count(x,formatted):
    qid2num = {}
    y = []
    for data_point in formatted:
        y.append(data_point.q_id)
    j = 0
    for i, qid in enumerate(np.unique(y)):
        qid2num[qid] = i
        print qid, " = ", i
        j = j + 1

    for i in range(0, 16):
        relevance = x.iloc[i * 1000 - 1000:i * 1000, 1]
        print i
        print collections.Counter(relevance)

#This is a rough plot across the dimension
def Rough_plot(formatted):

    vector_averages = []
    for data_point in formatted:
        vector_average = 0
        for dimension in data_point.features.featuresArray:
            vector_average += float(dimension)

        vector_average = vector_average / len(data_point.features.featuresArray)

        vector_averages.append(vector_average)

    plt.plot(vector_averages)
    plt.show()

# This plots all the dimension across documentation indexes across all 15000 datapoints
def plotAllD(formatted):
    for i in range(0,64):
        plot1D(formatted,i)
        plt.title(str(i+1))
        plt.show()


#this is used in plotAllD
def plot1D(formatted,f):
    # plot singular dimension
    Value = []
    for data_point in formatted:
        Value.append(data_point.features.featuresArray[f])
    plt.plot(Value)

#plot a vector into barchart
def plot_vector_barchart(plotarray):
    pc1 = plotarray
    N = len(pc1)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pc1, width, color='r')
    ax.set_xticks(ind + width / 2)

    numberString = []
    for i in range(1, 65):
        if i % 2 == 1:
            numberString.append(str(i))
        else:
            numberString.append('')
    ax.set_xticklabels(numberString)
    plt.show()

# now make Principle component analysis
def principle_component_analysis(formatted, features):

    q_ids = []
    for data_point in formatted:
        q_ids.append(data_point.q_id)

    # make qid numeric
    qid2num = {}
    j = 0
    for i, qid in enumerate(np.unique(q_ids)):
        qid2num[qid] = i
        print qid, " = ", i
        j = j + 1
    q_ids = [qid2num[qid] for qid in q_ids]
    # store in numpy arrays
    X = np.array(features)
    Y = np.array(q_ids)

    # fit PCA
    pca = PCA(64)
    pca.fit(X)
    X_2d = pca.transform(X)

    # plot PCA vector
    N = 64
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pc1, width, color='r')
    rects2 = ax.bar(ind + width, pc2, width, color='y')
    ax.legend((rects1[0], rects2[0]), ('pc1', 'pc2'))
    ax.set_xticks(ind + width / 2)

    numberString = []
    for i in range(1, 65):
        if i % 2 == 1:
            numberString.append(str(i))
        else:
            numberString.append('')
    ax.set_xticklabels(numberString)

    # plot decay of Eigenvalue

    fig, ax = plt.subplots()
    eigenvalue = pca.explained_variance_
    rects1 = ax.bar(ind, eigenvalue, width, color='b')

    print '1st 2pc = ', eigenvalue[0] + eigenvalue[1]
    eigensum = 0
    for i in range(2,64):
        eigensum = eigensum + eigenvalue[i]
    print 'remaining pcs = ', eigensum

    # plot first two principle components coloured by qid
    colours = sns.color_palette("hls", 15)
    colours = [colours[i] for i in q_ids]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], color=colours, alpha=0.6, s=2)
    ax.legend()
    ax.grid(True)
    plt.show()


# make covariance matrix, precision matrix by graphical lasso
def matrix_analysis(argv):
    x = pd.read_csv(argv[1], sep=',', header=None)
    features = x.iloc[:, 2:66].astype(np.float64)

    features = features.subtract(features.mean(axis=0), axis=1)
    features = features.divide(features.std(axis=0), axis=1)

    model = GraphLassoCV()
    model.fit(features)
    prec = model.get_precision()
    corr = features.corr()

    norm = SymLogNorm(linthresh=0.03, linscale=0.03,
                      vmin=-1.0, vmax=1.0)

    cmap = plt.get_cmap('PuOr')

    plt.imshow(corr, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.show()

    # plot precision matrix
    norm = SymLogNorm(linthresh=0.03, linscale=0.03,
                      vmin=-1.0, vmax=1.0)

    cmap = plt.get_cmap('PuOr')

    plt.imshow(prec, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.show()


    g = nx.Graph(prec)
    nx.draw_circular(g, with_labels=True)
    plt.show()

    ##########################################
    # Rearrange the colombs and plot again
    perm = [5, 6, 7, 8, 9]
    perm += [25, 26, 29, 30, 31, 34, 35, 36, 39]
    perm += [27, 28, 32, 33, 37, 38]
    perm += [61, 62, 63]
    perm += [20, 24, 40, 41]

    perm += [i for i in xrange(64) if i not in perm]
    features = features.iloc[:, perm]
    # print perm[27]
    model = GraphLassoCV()
    model.fit(features)
    prec = model.get_precision()
    corr = features.corr()

    #########cov
    features = features.iloc[:, perm]
    norm = SymLogNorm(linthresh=0.03, linscale=0.03,
                      vmin=-1.0, vmax=1.0)

    cmap = plt.get_cmap('PuOr')

    plt.imshow(corr, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    ############prec
    features = features.iloc[:, perm]
    norm = SymLogNorm(linthresh=0.03, linscale=0.03,
                      vmin=-1.0, vmax=1.0)

    cmap = plt.get_cmap('PuOr')

    plt.imshow(prec, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xticks([])
    plt.show()

#Logistic Regression
def logisticRe(formatted, features):
    y = []
    for data_point in formatted:
        y.append(data_point.relevance)

    # store in numpy arrays
    X = np.array(features)
    Y = np.array(y)


    model=LogisticRegression(C=1,penalty='l1', solver='liblinear')
    solution = cross_val(X, Y, model)    
    plot_vector_barchart(solution[1])
    print "score = ", solution[0]/5.0
    print solution[1]

#5 - fold cross-validation
def cross_val(X, y, model, v=5):

    l = len(X)

    total_score = 0.0
    for i in range(v):
        validate_mask = ((np.arange(l) % v) == i)
        training_mask = ((np.arange(l) % v) != i)
        model.fit(X[training_mask], y[training_mask])
        total_score += model.score(X[validate_mask], y[validate_mask])
    return total_score, model.coef_[0]


def serve_analysis_and_graph(formatted):
    app = dash.Dash()

    graphs = []
    id_mod = 1
    for data_set in data:
        # process_data_set(data_set)
        vector_averages = []
        for data_point in data_set:
            vector_average = 0
            for dimension in data_point.vector.data:
                vector_average += float(dimension)

            vector_average = vector_average / len(data_point.vector.data)

            vector_averages.append(vector_average)

        graphs.append(dcc.Graph(
            id='Vector Average ' + str(id_mod),
            figure={
                'data': [
                    {'x': range(1, len(vector_averages)), 'y': vector_averages, 'type': 'line',
                     'name': 'Vector Average' + str(id_mod)},
                ],
                'layout': {
                    'title': 'Average dimension value per data point'
                }
            }
        ))

    print len(graphs)

    children = [
        html.H1(children='Interactive Data Analysis'),

        html.Div(children='''
            Visualisation of data points
        '''),

        html.Div(children=graphs)
    ]

    # children = children + graphs
    app.layout = html.Div(children=children)
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

    app.run_server(debug=True)


def read_data_from_file(path):
    filename = path

    data = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            row = line.split(',')
            row.remove('')

            for i in xrange(len(row)):
                column = row[i]
                if column.find("\n") != -1:
                    row[i] = row[i].replace("\n", '')

            data_point = DataPoint(row)

            data.append(data_point)
            line = fp.readline()

    data = np.array(data)

    return data

#Class definition
class DataPoint:
    def __init__(self, data):
        self.relevance = data[0]
        self.features = VectorData(data[1:64])
        self.q_id = data[65]
        self.doc_id = data[66]


class VectorData:
    def __init__(self, data):
        self.featuresArray = np.array(data)

    def get_value(self, number):
        return self.data[number]


def string_to_hex(s):
    lst = []
    for ch in s:
        hv = hex(ord(ch)).replace('0x', '')
        if len(hv) == 1:
            hv = '0' + hv
        lst.append(hv)

    return int(reduce(lambda x, y: x + y, lst), 16)


def hex_to_string(s):
    if not isinstance(s, basestring):
        string = hex(s).rstrip("L").lstrip("0x")
    else:
        string = s
    return string and chr(atoi(string[:2], base=16)) + hex_to_string(string[2:]) or ''


if __name__ == '__main__':
    main(sys.argv)
