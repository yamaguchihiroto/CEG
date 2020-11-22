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


def derived_from_dirichlet(n,m,d,k,k2,alpha,beta,gamma,node_d,com_s,phi_d,phi_c,sigma_d,sigma_c,delta_d,delta_c,att_power,att_uniform,att_normal):
    def selectivity(pri=0,node_d=0,com_s=0):
        # priority
        priority_list = ["edge","degree"]
        priority = priority_list[pri]
        # node degree
        node_degree_list = ["power_law","uniform","normal","zipfian"]
        node_degree = node_degree_list[node_d]
        # community size
        com_size_list = ["power_law","uniform","normal","zipfian"]
        com_size = com_size_list[com_s]
        return priority, node_degree, com_size

    priority,node_degree,com_size = selectivity(node_d=node_d,com_s=com_s) # (("edge","degree"),("power_law","uniform","normal","zipfian"),("power_law","uniform","normal","zipfian"))

    # ## distribution generator

    def distribution_generator(flag, para_pow, para_normal, para_zip, t):
        if flag == "power_law":
#            dist = np.random.pareto(para_pow, t)
            dist = pareto.rvs(para_pow, size = t) - 1
            dist = dist / max(dist)

#            dist = 1 - np.random.power(para_pow, t) # R^{k}
#            dist = np.random.uniform(0,1,t)
#            dist = (dist ** para_pow)
        elif flag == "uniform":
            dist = np.random.uniform(0,1,t)
        elif flag == "normal":
            dist = np.random.normal(0.5,para_normal,t)
        elif flag == "zipfian":
            dist = np.random.zipf(para_zip,t)
        
        return dist

    # ## generate a community size list

    def community_generation(n, k, com_size, phi_c, sigma_c, delta_c):
        chi = distribution_generator(com_size, phi_c, sigma_c, delta_c, k)
        # chi = chi*(chi_max-chi_min)+chi_min
        chi = chi * alpha / sum(chi)
    #Cluster assignment
        U = np.random.dirichlet(chi, n) # topology cluster matrix R^{n*k}
        return U

    # ## Cluster assignment


    U = community_generation(n, k, com_size, phi_c, sigma_c, delta_c)

    # ## Edge construction

    # ## node degree generation

    def node_degree_generation(n, m, priority, node_degree, phi_d, sigma_d, delta_d):
        theta = distribution_generator(node_degree, phi_d, sigma_d, delta_d, n)        
        if priority == "edge":
            theta = np.array(list(map(int,theta * m * 2 / sum(theta) +1)))
        if max(theta) > n:
            raise Exception('Error! phi_d is too large.')

        # else:
        #     theta = np.array(list(map(int,theta*(theta_max-theta_min)+theta_min)))
        return theta


    theta = node_degree_generation(n, m, priority, node_degree, phi_d, sigma_d, delta_d)

    ## Attribute generation

    num_power = int(d*att_power)
    num_uniform = int(d*att_uniform)
    num_normal = int(d*att_normal)
    num_random = d-num_power-num_uniform-num_normal

    #Input4 for attribute

    beta_dist = "normal" # 0:power-law, 1:normal, 2:uniform
    gamma_dist = "normal" # 0:power-law, 1:normal, 2:uniform
    phi_V=2;sigma_V=0.1;delta_V=0.2
    phi_H=2;sigma_H=0.1;delta_H=0.2

    # generating V
    chi = distribution_generator(beta_dist, phi_V, sigma_V, delta_V, k2)
    chi=np.array(chi)/sum(chi)*beta
    V = np.random.dirichlet(chi, num_power+num_uniform+num_normal) # attribute cluster matrix R^{d*k2}
    # generating H
    chi = distribution_generator(gamma_dist, phi_H, sigma_H, delta_H, k2)
    chi=np.array(chi)/sum(chi)*gamma
    H = np.random.dirichlet(chi, k) # cluster transfer matrix R^{k*k2}

    return U,H,V,theta


def make_core(k, core_ratio, U, C, num_of_core):
    
    core_ex = []
    core_los = []
    core_node = []
#    print(core_node)
    core_count = np.zeros(k)
    for node_num, cluster_num in enumerate(C):
        if core_count[cluster_num] < core_ratio[cluster_num]:
            core_node.append(node_num)
            core_count[cluster_num] += 1
#    print("core node : ", core_count)
    for cluster_num, core_counter in enumerate(core_count):
        if core_counter < core_ratio[cluster_num]:
            print(f"{cluster_num} LOH")
    return core_node

def change(A, count_max, j_):
    t = A[-(count_max+1)]
    A[-(count_max+1)] = A[j_]
    A[j_] = t
    return A


def changeU(A, count_max, j_):
    t = np.array(A[-(count_max+1)])
    A[-(count_max+1)] = A[j_]
    for i in range(len(t)):
        A[j_][i] = t[i]
    return A




def CEG_acmark(outpath="",n=1000,m=10000 ,d=1,k=5,k2=10,alpha=0.1,beta=10,gamma=1,node_d=0,com_s=0,phi_d=100,phi_c=3,sigma_d=0.1,sigma_c=0.1,delta_d=3,delta_c=3,att_power=0.0,att_uniform=0.0,att_normal=1.0,att_ber=0.0,dev_normal_max=0.3,dev_normal_min=0.1,dev_power_max=3,dev_power_min=2,uni_att=0.2,core=[1]):
    if outpath == "":
        raise Exception('Error! outpath is emply.')
    if len(core) != k:
        raise Exception('Error! # cores list')
    U,H,V,theta = derived_from_dirichlet(n,m,d,k,k2,alpha,beta,gamma,node_d,com_s,phi_d,phi_c,sigma_d,sigma_c,delta_d,delta_c,att_power,att_uniform,att_normal)
    C = [] # cluster list (finally, R^{n})
    for i in range(n):
        C.append(np.argmax(U[i]))
    cluster_size = Counter(C)
    print("cluster size :", cluster_size)

    def edge_construction(n, U, C, theta, core, around = 1.0, r = 10*k):
        count_max = 0
        star_node = []
        max_core = 10
        core_ratio = np.zeros(k)
        num_of_core_in_clus = [0]*k
        for i in cluster_size.keys():
            num_of_core_in_clus[i] = core.pop(-1)
        for cluster_num, core in enumerate(num_of_core_in_clus):
            if cluster_num not in cluster_size:
                cluster_size[cluster_num] = 0
            if core > max_core:
                core_ratio[cluster_num] = 0
            else:
                core_ratio[cluster_num] = int(core)
        num_of_core = int(sum(core_ratio))
        
    # list of edge construction candidates
        S = sparse.dok_matrix((n,n))
        degree_list = np.zeros(n)
        theta = np.sort(theta)[::-1]
        core_node = make_core(k, core_ratio, U, C, num_of_core)
        candidate_nodes = list(range(n))
        for i in range(n):
            if i in core_node and core_ratio[C[i]] == 1 and theta[i] >= cluster_size[C[i]]:
                for node_num in range(i + 1,n):
                    if C[i] == C[node_num]:
                        S[i,node_num] = 1
                        S[node_num,i] = S[i,node_num]
                        star_node.append(node_num)
            count = 0
            if degree_list[i] == theta[i]:
                continue
            
            # step1 create candidate list
            if num_of_core_in_clus[C[i]] > max_core or i in core_node:#i >= 0 and i <= sum(core_ratio):#core_ratio[C[i]] == 0 or 
                candidate = candidate_nodes
            else:
                candidate = candidate_nodes[(int(len(candidate_nodes)/4)):]    
            if candidate == []:
                continue
                
            # step2 create edges
            candidate_order_dict = {}
            for j in candidate:
                if i < j:
                    i_ = i;j_ = j
                else:
                    i_ = j;j_ = i

                if i_ != j_ and S[i_,j_] == 0 and degree_list[i_] < around * theta[i_] and degree_list[j_] < around * theta[j_]:
                    candidate_order_dict[j_] = (1 - np.exp(-U[i_,:].transpose().dot(U[j_,:]))) * theta[j_]  # ingoring node degree
            while degree_list[i] < theta[i]:#count < r and degree_list[i] < theta[i]:
                if candidate_order_dict == {}:
                    break
                target_nodes = random.choices(list(candidate_order_dict.keys()), k=1, weights=np.power(np.array(list(candidate_order_dict.values())),1))
                target_node = int(target_nodes[0])
                if i in star_node and C[i] == C[target_node]:
                    del candidate_order_dict[target_node]
                    continue
                if S[i,target_node] != 1:
                    S[i,target_node] = 1
                    S[target_node,i] = S[i,target_node]
                    degree_list[i]+=1;degree_list[target_node]+=1
                    del candidate_order_dict[target_node]
                    if degree_list[target_node] == theta[target_node]:
                        candidate_nodes.remove(target_node)
                        count_max += 1
                    if degree_list[i] == theta[i]:
                        candidate_nodes.remove(i)
                        break



        return S

    S = edge_construction(n, U, C, theta, core)

    ### Attribute Generation ###

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    ### Construct attribute matrix X from latent factors ###
    X = U.dot(H.dot(V.T))
    # X = sigmoid(U.dot(H)).dot(W.transpose())

    # ### Variation of attribute
    num_bernoulli = int(d*att_ber)
    num_power = int(d*att_power)
    num_uniform = int(d*att_uniform)
    num_normal = int(d*att_normal)
    num_random = d-num_power-num_uniform-num_normal
    def variation_attribute(n,k,X,C,num_bernoulli,num_power,num_uniform,num_normal,dev_normal_min,dev_normal_max,dev_power_min,dev_power_max,uni_att):
        for i in range(num_bernoulli):
                for p in range(n):
                    X[p,i] = bernoulli.rvs(p=X[p,i], size=1)
        dim = num_bernoulli

        for i in range(num_power): # each demension
            clus_dev = np.random.uniform(dev_normal_min,dev_normal_max-0.1,k)
            exp = np.random.uniform(dev_power_min,dev_power_max,1)
            for p in range(n): # each node
                X[p,i] = (X[p,i] * np.random.normal(1.0,clus_dev[C[p]],1)) ** exp #clus_dev[C[p]]

        dim += num_power
        for i in range(dim,dim+num_uniform): # each demension
            clus_dev = np.random.uniform(1.0-uni_att,1.0+uni_att,n)
            for p in range(n):
                X[p,i] *= clus_dev[C[p]]

        dim += num_uniform
        for i in range(dim,dim+num_normal): # each demension
            clus_dev = np.random.uniform(dev_normal_min,dev_normal_max,k)
            for p in range(n): # each node
                X[p,i] *= np.random.normal(1.0,clus_dev[C[p]],1)
        return X

    ### Apply probabilistic distributions to X ###

    X = variation_attribute(n,k,X,C,num_bernoulli,num_power,num_uniform,num_normal,dev_normal_min,dev_normal_max,dev_power_min,dev_power_max,uni_att)

    # random attribute
    def concat_random_attribute(n,X,num_random):
        rand_att = []
        for i in range(num_random):
            random_flag = random.randint(0,2)
            if random_flag == 0:
                rand_att.append(np.random.normal(0.5,0.2,n))
            elif random_flag == 1:
                rand_att.append(np.random.uniform(0,1,n))
            else:
                rand_att.append(1.0-np.random.power(2,n))
        return np.concatenate((X, np.array(rand_att).T), axis=1)

    if num_random != 0:
        X = concat_random_attribute(n,X,num_random)

    # ## Regularization for attributes

    for i in range(d):
        X[:,i] -= np.amin(X[:,i])
        X[:,i] /= np.amax(X[:,i])

    sio.savemat(outpath,{'S':S,'X':X,'C':C})
