#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:57:36 2020

@author: yamaguchihiroto
"""

from scipy import io
import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import sparse
import random
import scipy.io as sio
from scipy.stats import bernoulli
import plotly
from collections import Counter
import time
from collections import defaultdict
from scipy.stats import pareto


def div_clus(all_adj_list, num_of_cluster, C):
    
    cluster_adj_list = []
    cluster_G_list = []
    intra_edge = 0
    inter_edge = 0
    intra_edge_par = defaultdict(int)
    for cluster_num in range(num_of_cluster):
        adj_list_C = []
        for j, edge in enumerate(all_adj_list):
            if C[edge[0]] == cluster_num and C[edge[1]] == cluster_num:
                adj_list_C.append(edge)
                intra_edge += 1
                intra_edge_par[edge[0]] += 1
                intra_edge_par[edge[1]] += 1
        cluster_adj_list.append(adj_list_C)
        G = nx.Graph()
        G.add_edges_from(cluster_adj_list[cluster_num])

#         if G.number_of_nodes() <= 10:
#             continue

        cluster_G_list.append(G)
    
    return cluster_adj_list, cluster_G_list, intra_edge_par


def hub_dom_CCF(cluster_G_list, num_of_cluster):
    hub_dominance_list = np.zeros(num_of_cluster)
    CCF_list = np.zeros(num_of_cluster)
    for cluster_num, cluster_G in enumerate(cluster_G_list):
        degree_list = list(dict(nx.degree(cluster_G)).values())
        if degree_list == []:
            continue
        max_degree = max(degree_list)
        node = cluster_G.number_of_nodes()        
        hub_dominance = max_degree / (node - 1)
        hub_dominance_list[cluster_num] = hub_dominance
        CCF_list[cluster_num] = nx.average_clustering(cluster_G)

    return hub_dominance_list, CCF_list




def node_degree(G_all, file_name):
    plt.hist(dict(nx.degree(G_all)).values(), bins = 19)
    plt.savefig(f"{file_name}.png")
    degree1 = list(dict(nx.degree(G_all)).values())
    n1 = len(degree1)
    nume = 0
    for i in range(n1):
        nume = nume + (i+1)*degree1[i]
    
    deno = n1*sum(degree1)
    gini = ((2*nume)/deno - (n1+1)/(n1))*(n1/(n1-1))
    print("degree_gini : " + str(gini))




def all_graph_adj(S):
    adj_list = []
    S = S.tocoo()
    S_len = S.getnnz()
    for i in range(S_len):
        adj_list.append([S.row[i],S.col[i]])
    adj_list = np.array(adj_list)
    G = nx.Graph()
    G.add_edges_from(adj_list)

    return G, adj_list

def cluster_graph_adj(cluster_adj_list,file_name):
    with open(f'{file_name}.csv', 'w') as csvFile1:
        writer = csv.writer(csvFile1,delimiter=",")
        for data_list in cluster_adj_list[0]:
            writer.writerow(data_list)


def CCF_Hub_cluster_size(CCF_list, hub_dominance_list, cluster_size,file_name):
    layout = plotly.graph_objs.Layout(
    title=f"{file_name}",
    scene=plotly.graph_objs.Scene(
        xaxis=plotly.graph_objs.layout.scene.XAxis(title = "CCF", range=[0, 1]),
        yaxis=plotly.graph_objs.layout.scene.YAxis(title = "Hub dominance", range=[0, 1]),
        zaxis=plotly.graph_objs.layout.scene.ZAxis(title = "cluster size", rangemode='tozero'),
        )
    )
    trace = plotly.graph_objs.Scatter3d(x = CCF_list, y = hub_dominance_list, z = cluster_size, mode = 'markers')
    data = [trace]
    fig = plotly.graph_objs.Figure(data=data, layout=layout) 
    plot_url = plotly.offline.plot(fig, auto_open = False, filename=f"{file_name}.html")


def CCF_Hub_heatmap(CCF_list,hub_dominance_list, file_name):
    CCF_list_ave = sum(CCF_list) / len(CCF_list)
    hub_dominance_list_ave = sum(hub_dominance_list) / len(hub_dominance_list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.rcParams["font.size"] = 16
    H = ax.hist2d(CCF_list,hub_dominance_list, bins=[np.linspace(0,1,40),np.linspace(0,1,40)],cmap="Reds")
    ax.set_xlabel('CCF',fontsize = 18)
    ax.set_ylabel('hub_dom',fontsize = 18)
    fig.colorbar(H[3],ax=ax)
    plt.savefig(f"{file_name}.png")
    plt.show()
    
    
def analysis(file_name):
    hub_dominance_list = []
    CCF_list = []
    cluster_size_list = []
    intra_edge_list = []
    matdata = io.loadmat(f"{file_name}.mat", squeeze_me=True)
    X = matdata["X"]
    S = matdata["S"]
    C = matdata["C"]
    G_all, all_adj_list = all_graph_adj(S)
    num_of_cluster = max(C) + 1
    
    cluster_adj_list, cluster_G_list, intra_edge_par = div_clus(all_adj_list, num_of_cluster, C)
    num_of_cluster = len(cluster_G_list)
    intra_edges = 0
    degree = defaultdict(int)
    for node in intra_edge_par.keys():
        intra_edge_par[node] = intra_edge_par[node] / G_all.degree(node) / 2
        degree[node] = G_all.degree(node)

    for cluster in cluster_G_list:
        num_of_edges = cluster.number_of_edges()
        intra_edges += num_of_edges
    intra_edge_list.append(intra_edges)

    
    hub_domi, CCFs = hub_dom_CCF(cluster_G_list, num_of_cluster)
    for j in hub_domi:
        hub_dominance_list.append(j)
    for j in CCFs:
        CCF_list.append(j)
    for cluster_G in cluster_G_list:
        cluster_size_list.append(cluster_G.number_of_nodes())


    #print(hub_domi, CCFs)
    #cluster_graph_adj(cluster_adj_list)
#    print(sum(hub_dominance_list)/len(hub_dominance_list), sum(CCF_list)/len(CCF_list),sum(intra_edge_list)/len(intra_edge_list))
#    node_degree(G_all,file_name)
    CCF_Hub_heatmap(CCF_list,hub_dominance_list,file_name)
#    CCF_Hub_cluster_size(CCF_list, hub_dominance_list, cluster_size_list,file_name)
