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
#            elif C[edge[0]] == cluster_num and C[edge[1]] != cluster_num or C[edge[0]] != cluster_num and C[edge[1]] == cluster_num:
#                inter_edge += 1
#        print(adj_list_C)
        cluster_adj_list.append(adj_list_C)
        G = nx.Graph()
        G.add_edges_from(cluster_adj_list[cluster_num])
#        nx.draw_networkx(G)
#        plt.show()

        if G.number_of_nodes() <= 10:
            continue
        cluster_G_list.append(G)
#    print(intra_edge / 2 + inter_edge / 4)
#    adj_list2 = np.array(adj_list2)
    return cluster_adj_list, cluster_G_list, intra_edge_par


def hub_dom_CCF(cluster_G_list, num_of_cluster):
    hub_dominance_list = np.zeros(num_of_cluster)
    CCF_list = np.zeros(num_of_cluster)
    
    
    for cluster_num, cluster_G in enumerate(cluster_G_list):
        degree_list = list(dict(nx.degree(cluster_G)).values())
#        print(degree_list)        
        if degree_list == []:
            continue
        
        max_degree = max(degree_list)

        node = cluster_G.number_of_nodes()        

        hub_dominance = max_degree / (node - 1)
        hub_dominance_list[cluster_num] = hub_dominance
        
        
        CCF_list[cluster_num] = nx.average_clustering(cluster_G)

    return hub_dominance_list, CCF_list




def node_degree(G_all, file_name):
    plt.hist(dict(nx.degree(G_all)).values(), bins = 19)#np.logspace(0,3,19), log = False)
#    plt.gca().set_xscale("log")
#    plt.ylim(1,10000)
    plt.savefig(f"node_degree_pic/{file_name}_upto10_font_18.png")
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
#    print(S_len)
    for i in range(S_len):
        adj_list.append([S.row[i],S.col[i]])
    adj_list = np.array(adj_list)
#    print(adj_list)
    G = nx.Graph()
    G.add_edges_from(adj_list)

#    with open('move_mac/adjacency_list_pl_all_heat_25000_0.1_PA.csv', 'w') as csvFile1:
#        writer = csv.writer(csvFile1,delimiter=",")
#        for data_list in adj_list:
#            writer.writerow(data_list)

    return G, adj_list

def cluster_graph_adj(cluster_adj_list,file_name):
    with open(f'move_mac/{file_name}_upto10_upto10_font_18.csv', 'w') as csvFile1:
        writer = csv.writer(csvFile1,delimiter=",")
        for data_list in cluster_adj_list[0]:
            writer.writerow(data_list)


def CCF_Hub_cluster_size(CCF_list, hub_dominance_list, cluster_size,file_name):
    layout = plotly.graph_objs.Layout(
    title=f"{file_name}_upto10_font_18",
    scene=plotly.graph_objs.Scene(
        xaxis=plotly.graph_objs.layout.scene.XAxis(title = "CCF", range=[0, 1]),
        yaxis=plotly.graph_objs.layout.scene.YAxis(title = "Hub dominance", range=[0, 1]),
        zaxis=plotly.graph_objs.layout.scene.ZAxis(title = "cluster size", rangemode='tozero'),
        )
    )
    trace = plotly.graph_objs.Scatter3d(x = CCF_list, y = hub_dominance_list, z = cluster_size, mode = 'markers')
    data = [trace]
    fig = plotly.graph_objs.Figure(data=data, layout=layout) 
    plot_url = plotly.offline.plot(fig, auto_open = False, filename=f"CCF_Hub_cluster_size/{file_name}_upto10_font_18.html")


def CCF_Hub_heatmap(CCF_list,hub_dominance_list, file_name):
    CCF_list_ave = sum(CCF_list) / len(CCF_list)
    hub_dominance_list_ave = sum(hub_dominance_list) / len(hub_dominance_list)
#    print(CCF_list_ave,hub_dominance_list_ave)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.rcParams["font.size"] = 16
    H = ax.hist2d(CCF_list,hub_dominance_list, bins=[np.linspace(0,1,40),np.linspace(0,1,40)],cmap="Reds")
    ax.set_xlabel('CCF',fontsize = 18)
    ax.set_ylabel('hub_dom',fontsize = 18)
    fig.colorbar(H[3],ax=ax)
    plt.savefig(f"heatmap_pic/{file_name}_upto10_font_18_Reds.png")
    plt.show()