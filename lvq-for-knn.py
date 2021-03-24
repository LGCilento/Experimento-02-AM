import pandas as pd
import numpy as np
#from google.colab import files
from sklearn import preprocessing
from scipy.io import arff
from sklearn.utils import shuffle
from statistics import mode
from statistics import mean 
import code
import matplotlib.pyplot as plt
import math as mt

def euclideanDistance(x, y):
    dist = (np.linalg.norm(x-y))
    return dist

def knn_normal(k, train, test, test_labels):
    predicted_values = []
    train_aux = train.drop(columns=['problems'])
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    total = test.shape[0]
    for l in test.index:
        i = test.iloc[l]
        i_neibours = []
        for j in train_aux.index:
            neibour = (train.iloc[j]["problems"],
                       euclideanDistance(i, train_aux.iloc[j]))
            i_neibours.append(neibour)
        i_neibours = sorted(i_neibours, key=lambda x: x[1])
        get_nearst_neibours = i_neibours[:k]
        get_nearst_neibours_values = [a for a, b in get_nearst_neibours]
        classification = int(mode(get_nearst_neibours_values))
        predicted_values.append(classification)
        real_value = test_labels[l]
        if (classification > 0 and real_value > 0):
            true_positive += 1
        elif classification > 0 and real_value == 0:
            false_positive += 1
        elif classification == 0 and real_value == 0:
            true_negative += 1
        elif classification == 0 and real_value > 0:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative #,predicted_values

def generate_prototypes(train,n):
    amount_negative_prototypes = mt.ceil(n/2)           # Quantidade de protótipos da classe negativa
    amount_positive_prototypes = mt.floor(n/2)          # Quantidade de protótipos da classe positiva 
    train_positive = train.loc[train['problems'] > 0]   # Partição do conjunto de treinamento que contém as classes positivas
    train_negative = train.loc[train['problems'] == 0]  # Partição do conjunto de treinamento que contém as classes negativas

    negative_prototypes = []
    positive_prototypes = []
    class_positive_centroid = train_positive.mean()
    class_negative_centroid = train_negative.mean()
    var_max_positive = train_positive.var()/20
    var_min_positive = train_positive.var()/100
    var_max_negative = train_negative.var()/20
    var_min_negative = train_negative.var()/100

    for i in range(0,amount_negative_prototypes):

        if i == 0:
            negative_prototypes.append(class_negative_centroid)
        else:
            new_prototype = class_negative_centroid
            for i in train_negative.columns:
                if i == 'problems' or i == 'defects':
                    break
                else:
                    new_prototype[i] = new_prototype[i] + np.random.uniform(var_min_negative[i],var_max_negative[i]) 
            negative_prototypes.append( new_prototype )
            

    for i in range(0,amount_positive_prototypes):

        if i == 0:
            positive_prototypes.append(class_positive_centroid)
        else:
            new_prototype = class_positive_centroid
            for i in train_positive.columns:
                if i == 'problems' or i == 'defects':
                    break
                else:
                    new_prototype[i] = new_prototype[i] + np.random.uniform(var_min_positive[i],var_max_positive[i])            
            positive_prototypes.append( new_prototype )

    return positive_prototypes, negative_prototypes 

def get_nearst_prototype(x,prototypes):
    min_dist = 9999999999999
    index = 0
    j = 0
    for i in prototypes:
        dist = euclideanDistance(x,i)
        if dist < min_dist:
            min_dist = dist
            index = j
        j += 1
    return prototypes[index],index

def get_ordered_nearst_prototypes(x,prototypes):    # Esta função aceita uma lista de tuplas (a,b) onde a = protótipo e b = indíce do protótipo
    ordered_prototypes = []

    for p in prototypes:
        dist = euclideanDistance(x,p[0])
        ordered_prototypes.append((p[0],p[1],dist))
    ordered_prototypes = sorted(ordered_prototypes,key= lambda f: f[2])

    return ordered_prototypes                       # retorna uma lista de tuplas (a,b,c) onde a,b já foram descritos e c é a distância do protótipo para a instância x

def get_class(x):
    if x['problems'] > 0:
        return True
    else:
        return False

def get_index_prototypes(prototypes):
    j = 0
    indexed_prototypes = []
    for i in prototypes:
        indexed_prototypes.append((i,j))
        j += 1
    return indexed_prototypes

def inside_window(p1,p2):
    w = 0.2
    s = (1 - w) / (1 + w)
    if min(p1,p2) > s:
        return True
    else:
        return False

def lvq1(df,positive_prototypes,negative_prototypes,n):
    #prototypes = generate_prototypes(df,n)
    #df.reset_index(drop=True)
    prototypes = positive_prototypes[0:mt.floor(n/2)] + negative_prototypes[0:mt.ceil(n/2)]
    for i in range(0,df.shape[0]):
        try: df.iloc[i]
        except: code.interact(local=dict(globals(), **locals()))
        else:
            x = df.iloc[i]
            nearst_prototype,index = get_nearst_prototype(x,prototypes)

            if get_class(nearst_prototype) == get_class(x):
                nearst_prototype = nearst_prototype + 0.2 * (x - nearst_prototype)
            else:
                nearst_prototype = nearst_prototype - 0.2 * (x - nearst_prototype)
            prototypes[index] = nearst_prototype
    return pd.DataFrame(prototypes)

def lvq2(df,positive_prototypes,negative_prototypes,n):
    #df.reset_index(drop=True)
    #prototypes = generate_prototypes(df,n)
    prototypes = positive_prototypes[0:mt.floor(n/2)] + negative_prototypes[0:mt.ceil(n/2)]
    indexed_prototypes = get_index_prototypes(prototypes)
    for i in range(0,df.shape[0]):
        try: df.iloc[i]
        except: code.interact(local=dict(globals(), **locals()))
        else:
            x = df.iloc[i]
            nearst_prototypes = get_ordered_nearst_prototypes(x,indexed_prototypes)
            nearst_prototype1 = nearst_prototypes[0][0]
            nearst_prototype2 = nearst_prototypes[1][0]
            nearst_prototype1_index = nearst_prototypes[0][1]
            nearst_prototype2_index = nearst_prototypes[1][1]
            dist_x_p1 = nearst_prototypes[0][2]
            dist_x_p2 = nearst_prototypes[1][2]


            if  inside_window(dist_x_p1,dist_x_p2):
                if (get_class(nearst_prototype1) != get_class(nearst_prototype2)):
                    if get_class(nearst_prototype1) == get_class(x):
                        nearst_prototype1 = nearst_prototype1 + 0.2 * (x - nearst_prototype1)
                        nearst_prototype2 = nearst_prototype2 - 0.2 * (x - nearst_prototype2)
                    else:
                        nearst_prototype1 = nearst_prototype1 - 0.2 * (x - nearst_prototype1)
                        nearst_prototype2 = nearst_prototype2 + 0.2 * (x - nearst_prototype2)

            for p in indexed_prototypes:
                if p[1] == nearst_prototype1_index:
                    prototypes[p[1]] = nearst_prototype1
                if p[1] == nearst_prototype2_index:
                    prototypes[p[1]] = nearst_prototype2
    lvq2_prototypes = pd.DataFrame(prototypes)
    lvq2_prototypes = lvq2_prototypes.reset_index(drop=True)

    return lvq2_prototypes

def lvq3(df,positive_prototypes,negative_prototypes,n):
    #df.reset_index(drop=True)
    #prototypes = generate_prototypes(df,n)
    prototypes = positive_prototypes[0:mt.floor(n/2)] + negative_prototypes[0:mt.ceil(n/2)]
    e = 0.3
    indexed_prototypes = get_index_prototypes(prototypes)
    for i in range(0,df.shape[0]):
        try: df.iloc[i]
        except: code.interact(local=dict(globals(), **locals()))
        else:
            x = df.iloc[i]
            nearst_prototypes = get_ordered_nearst_prototypes(x,indexed_prototypes)
            nearst_prototype1 = nearst_prototypes[0][0]
            nearst_prototype2 = nearst_prototypes[1][0]
            nearst_prototype1_index = nearst_prototypes[0][1]
            nearst_prototype2_index = nearst_prototypes[1][1]
            dist_x_p1 = nearst_prototypes[0][2]
            dist_x_p2 = nearst_prototypes[1][2]


            if  inside_window(dist_x_p1,dist_x_p2):
                if (get_class(nearst_prototype1) != get_class(nearst_prototype2)):
                    if get_class(nearst_prototype1) == get_class(x):
                        nearst_prototype1 = nearst_prototype1 + 0.2 * (x - nearst_prototype1)
                        nearst_prototype2 = nearst_prototype2 - 0.2 * (x - nearst_prototype2)
                    
                else:
                    nearst_prototype1 = nearst_prototype1 + e * 0.2 * (x - nearst_prototype1)
                    nearst_prototype2 = nearst_prototype2 + e * 0.2 * (x - nearst_prototype2)

            for p in indexed_prototypes:
                if p[1] == nearst_prototype1_index:
                    prototypes[p[1]] = nearst_prototype1
                if p[1] == nearst_prototype2_index:
                    prototypes[p[1]] = nearst_prototype2
    lvq3_prototypes = pd.DataFrame(prototypes)
    lvq3_prototypes = lvq3_prototypes.reset_index(drop=True)

    return lvq3_prototypes

def execute_experiment(df_in,k_values,amount_prototypes):
    x = df_in.values                        
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
   
    df = shuffle(df)
    df.columns = df_in.columns
    df_divided_list = np.array_split(df, 5)
    tp_rate_list_default = []
    fp_rate_list_default = []
    tp_rate_list_lvq1 = []
    fp_rate_list_lvq1 = []
    tp_rate_list_lvq2 = []
    fp_rate_list_lvq2 = []
    tp_rate_list_lvq3 = []
    fp_rate_list_lvq3 = []
    
    df_aux = df_divided_list.copy()
    test = df_aux[0]
    df_aux.pop(0)
    train_base = pd.concat(df_aux)

    for i in range(0,5):
        df_aux = df_divided_list.copy()
        test = df_aux[i]
        df_aux.pop(i)
        train_base = pd.concat(df_aux)
        tp_rate_list = []
        fp_rate_list = []
        for k in k_values:
            true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate = pre_knn(train_base,test,k)
            #print("k-NN DEFAULD: K = {}:  TP = {}, FP = {}, TN = {}, FN = {}, TP_RATE = {}, FP_RATE = {}".format(k, true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate) )
            tp_rate_list_default.append((k,tp_rate))
            fp_rate_list_default.append((k,fp_rate))
            
            lvq_positive_prototypes,lvq_negative_prototypes = generate_prototypes(train_base,10)
            #lvq_prototypes = lvq_positive_prototypes + lvq_negative_prototypes
            #code.interact(local=dict(globals(), **locals()))
            for n in amount_prototypes:
                train_lvq1 = lvq1(train_base,lvq_positive_prototypes,lvq_negative_prototypes,n)
                true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate = pre_knn(train_lvq1,test,k) # add conf_matrix_lvq1
                tp_rate_list.append((k,n,tp_rate))
                fp_rate_list.append((k,n,fp_rate))
                #print("k-NN with LVQ1: K = {}:  TP = {}, FP = {}, TN = {}, FN = {}, TP_RATE = {}, FP_RATE = {} (N-prototypes = {})".format(k, true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate, n) )
            #print("LVQ1: tp_rate_list_lvq1 = {}".format(tp_rate_list))
            #print("LVQ1: fp_rate_list_lvq1 = {}".format(fp_rate_list))
            tp_rate_list_lvq1.append(tp_rate_list)
            fp_rate_list_lvq1.append(fp_rate_list)
            tp_rate_list = []
            fp_rate_list = []
            for n in amount_prototypes:
                train_lvq2 = lvq2(train_base,lvq_positive_prototypes,lvq_negative_prototypes,n)
                true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate = pre_knn(train_lvq2, test, k) # # add conf_matrix_lvq2
                tp_rate_list.append((k,n,tp_rate))
                fp_rate_list.append((k,n,fp_rate))
                #print("k-NN with LVQ2.1: K = {}:  TP = {}, FP = {}, TN = {}, FN = {}, TP_RATE = {}, FP_RATE = {} (N-prototypes = {})".format(k, true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate, n) )
            #print("LVQ2.1: tp_rate_list_lvq2 = {}".format(tp_rate_list))
            #print("LVQ2.1: fp_rate_list_lvq2 = {}".format(fp_rate_list))
            tp_rate_list_lvq2.append(tp_rate_list)
            fp_rate_list_lvq2.append(fp_rate_list)
            tp_rate_list = []
            fp_rate_list = []
            for n in amount_prototypes:
                train_lvq3 =lvq3(train_base,lvq_positive_prototypes,lvq_negative_prototypes,n)
                true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate = pre_knn(train_lvq3, test, k) # add conf_matrix_lvq3
                tp_rate_list.append((k,n,tp_rate))
                fp_rate_list.append((k,n,fp_rate))
                #print("k-NN with LVQ3: K = {}:  TP = {}, FP = {}, TN = {}, FN = {}, TP_RATE = {}, FP_RATE = {} (N-prototypes = {})".format(k, true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate, n) )
            #print("LVQ3: tp_rate_list_lvq3 = {}".format(tp_rate_list))
            #print("LVQ3: fp_rate_list_lvq3 = {}".format(fp_rate_list))
            tp_rate_list_lvq3.append(tp_rate_list)
            fp_rate_list_lvq3.append(fp_rate_list)
            tp_rate_list = []
            fp_rate_list = []
        #print(tp_rate_list_default)
    print_results(tp_rate_list_default,fp_rate_list_default,tp_rate_list_lvq1,fp_rate_list_lvq1,tp_rate_list_lvq2,fp_rate_list_lvq2,tp_rate_list_lvq3,fp_rate_list_lvq3)

def print_results(tp_rate_list_default,fp_rate_list_default,tp_rate_list_lvq1,fp_rate_list_lvq1,tp_rate_list_lvq2,fp_rate_list_lvq2,tp_rate_list_lvq3,fp_rate_list_lvq3):
    knn_default_1_tp = [] ## k-nn sem LVQ
    knn_default_1_fp = []
    knn_default_3_tp = []
    knn_default_3_fp = []

    knn_lvq1_n2_1_tp = [] ## k-nn com LVQ1 n = 2
    knn_lvq1_n2_1_fp = []
    knn_lvq1_n2_3_tp = []
    knn_lvq1_n2_3_fp = []

    knn_lvq1_n4_1_tp = [] ## k-nn com LVQ1 n = 4
    knn_lvq1_n4_1_fp = []
    knn_lvq1_n4_3_tp = []
    knn_lvq1_n4_3_fp = []

    knn_lvq1_n6_1_tp = [] ## k-nn com LVQ1 n = 6
    knn_lvq1_n6_1_fp = []
    knn_lvq1_n6_3_tp = []
    knn_lvq1_n6_3_fp = []

    knn_lvq1_n8_1_tp = [] ## k-nn com LVQ1 n = 8
    knn_lvq1_n8_1_fp = []
    knn_lvq1_n8_3_tp = []
    knn_lvq1_n8_3_fp = []

    knn_lvq1_n10_1_tp = [] ## k-nn com LVQ1 n = 10
    knn_lvq1_n10_1_fp = []
    knn_lvq1_n10_3_tp = []
    knn_lvq1_n10_3_fp = []

    knn_lvq2_n2_1_tp = [] ## k-nn com LVQ2.1 n = 2
    knn_lvq2_n2_1_fp = []
    knn_lvq2_n2_3_tp = []
    knn_lvq2_n2_3_fp = []

    knn_lvq2_n4_1_tp = [] ## k-nn com LVQ2.1 n = 4
    knn_lvq2_n4_1_fp = []
    knn_lvq2_n4_3_tp = []
    knn_lvq2_n4_3_fp = []

    knn_lvq2_n6_1_tp = [] ## k-nn com LVQ2.1 n = 6
    knn_lvq2_n6_1_fp = []
    knn_lvq2_n6_3_tp = []
    knn_lvq2_n6_3_fp = []

    knn_lvq2_n8_1_tp = [] ## k-nn com LVQ2.1 n = 8
    knn_lvq2_n8_1_fp = []
    knn_lvq2_n8_3_tp = []
    knn_lvq2_n8_3_fp = []

    knn_lvq2_n10_1_tp = [] ## k-nn com LVQ2.1 n = 10
    knn_lvq2_n10_1_fp = []
    knn_lvq2_n10_3_tp = []
    knn_lvq2_n10_3_fp = []

    knn_lvq3_n2_1_tp = [] ## k-nn com LVQ3 n = 2
    knn_lvq3_n2_1_fp = []
    knn_lvq3_n2_3_tp = []
    knn_lvq3_n2_3_fp = []

    knn_lvq3_n4_1_tp = [] ## k-nn com LVQ3 n = 4
    knn_lvq3_n4_1_fp = []
    knn_lvq3_n4_3_tp = []
    knn_lvq3_n4_3_fp = []

    knn_lvq3_n6_1_tp = [] ## k-nn com LVQ3 n = 6
    knn_lvq3_n6_1_fp = []
    knn_lvq3_n6_3_tp = []
    knn_lvq3_n6_3_fp = []

    knn_lvq3_n8_1_tp = [] ## k-nn com LVQ3 n = 8
    knn_lvq3_n8_1_fp = []
    knn_lvq3_n8_3_tp = []
    knn_lvq3_n8_3_fp = []

    knn_lvq3_n10_1_tp = [] ## k-nn com LVQ3 n = 10
    knn_lvq3_n10_1_fp = []
    knn_lvq3_n10_3_tp = []
    knn_lvq3_n10_3_fp = []


    for i in tp_rate_list_default:      # kNN Default
        if i[0] == 1:
            knn_default_1_tp.append(i[1])
        else:
            knn_default_3_tp.append(i[1])

    for i in fp_rate_list_default:
        if i[0] == 1:
            knn_default_1_fp.append(i[1])
        else:
            knn_default_3_fp.append(i[1])

    for i in tp_rate_list_lvq1:                 # kNN comLVQ1
        for j in i:
            if j[0] == 1:
                if j[1] == 2:
                    knn_lvq1_n2_1_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq1_n4_1_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq1_n6_1_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq1_n8_1_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq1_n10_1_tp.append(j[2])
            else:
                if j[1] == 2:
                    knn_lvq1_n2_3_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq1_n4_3_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq1_n6_3_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq1_n8_3_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq1_n10_3_tp.append(j[2])

    for i in fp_rate_list_lvq1:
            for j in i:
                if j[0] == 1:
                    if j[1] == 2:
                        knn_lvq1_n2_1_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq1_n4_1_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq1_n6_1_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq1_n8_1_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq1_n10_1_fp.append(j[2])
                else:
                    if j[1] == 2:
                        knn_lvq1_n2_3_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq1_n4_3_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq1_n6_3_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq1_n8_3_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq1_n10_3_fp.append(j[2])

    for i in tp_rate_list_lvq2:                 # kNN comLVQ2.1
        for j in i:
            if j[0] == 1:
                if j[1] == 2:
                    knn_lvq2_n2_1_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq2_n4_1_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq2_n6_1_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq2_n8_1_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq2_n10_1_tp.append(j[2])
            else:
                if j[1] == 2:
                    knn_lvq2_n2_3_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq2_n4_3_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq2_n6_3_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq2_n8_3_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq2_n10_3_tp.append(j[2])

    for i in fp_rate_list_lvq2:   # LVQ2.1
            for j in i:
                if j[0] == 1:
                    if j[1] == 2:
                        knn_lvq2_n2_1_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq2_n4_1_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq2_n6_1_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq2_n8_1_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq2_n10_1_fp.append(j[2])
                else:
                    if j[1] == 2:
                        knn_lvq2_n2_3_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq2_n4_3_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq2_n6_3_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq2_n8_3_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq2_n10_3_fp.append(j[2])

    for i in tp_rate_list_lvq3:                 # kNN comLVQ3
        for j in i:
            if j[0] == 1:
                if j[1] == 2:
                    knn_lvq3_n2_1_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq3_n4_1_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq3_n6_1_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq3_n8_1_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq3_n10_1_tp.append(j[2])
            else:
                if j[1] == 2:
                    knn_lvq3_n2_3_tp.append(j[2])
                elif j[1] == 4:
                    knn_lvq3_n4_3_tp.append(j[2])
                elif j[1] == 6:
                    knn_lvq3_n6_3_tp.append(j[2])
                elif j[1] == 8:
                    knn_lvq3_n8_3_tp.append(j[2])
                elif j[1] == 10:
                    knn_lvq3_n10_3_tp.append(j[2])

    for i in fp_rate_list_lvq3:   # LVQ3
            for j in i:
                if j[0] == 1:
                    if j[1] == 2:
                        knn_lvq3_n2_1_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq3_n4_1_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq3_n6_1_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq3_n8_1_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq3_n10_1_fp.append(j[2])
                else:
                    if j[1] == 2:
                        knn_lvq3_n2_3_fp.append(j[2])
                    elif j[1] == 4:
                        knn_lvq3_n4_3_fp.append(j[2])
                    elif j[1] == 6:
                        knn_lvq3_n6_3_fp.append(j[2])
                    elif j[1] == 8:
                        knn_lvq3_n8_3_fp.append(j[2])
                    elif j[1] == 10:
                        knn_lvq3_n10_3_fp.append(j[2])

    lvq1_k1_tp_values = [mean(knn_lvq1_n2_1_tp),mean(knn_lvq1_n4_1_tp),mean(knn_lvq1_n6_1_tp),mean(knn_lvq1_n8_1_tp),mean(knn_lvq1_n10_1_tp)]
    lvq1_k3_tp_values = [mean(knn_lvq1_n2_3_tp),mean(knn_lvq1_n4_3_tp),mean(knn_lvq1_n6_3_tp),mean(knn_lvq1_n8_3_tp),mean(knn_lvq1_n10_3_tp)]
    lvq1_k1_fp_values = [mean(knn_lvq1_n2_1_fp),mean(knn_lvq1_n4_1_fp),mean(knn_lvq1_n6_1_fp),mean(knn_lvq1_n8_1_fp),mean(knn_lvq1_n10_1_fp)]
    lvq1_k3_fp_values = [mean(knn_lvq1_n2_3_fp),mean(knn_lvq1_n4_3_fp),mean(knn_lvq1_n6_3_fp),mean(knn_lvq1_n8_3_fp),mean(knn_lvq1_n10_3_fp)]

    lvq2_k1_tp_values = [mean(knn_lvq2_n2_1_tp),mean(knn_lvq2_n4_1_tp),mean(knn_lvq2_n6_1_tp),mean(knn_lvq2_n8_1_tp),mean(knn_lvq2_n10_1_tp)]
    lvq2_k3_tp_values = [mean(knn_lvq2_n2_3_tp),mean(knn_lvq2_n4_3_tp),mean(knn_lvq2_n6_3_tp),mean(knn_lvq2_n8_3_tp),mean(knn_lvq2_n10_3_tp)]
    lvq2_k1_fp_values = [mean(knn_lvq2_n2_1_fp),mean(knn_lvq2_n4_1_fp),mean(knn_lvq2_n6_1_fp),mean(knn_lvq2_n8_1_fp),mean(knn_lvq2_n10_1_fp)]
    lvq2_k3_fp_values = [mean(knn_lvq2_n2_3_fp),mean(knn_lvq2_n4_3_fp),mean(knn_lvq2_n6_3_fp),mean(knn_lvq2_n8_3_fp),mean(knn_lvq2_n10_3_fp)]

    lvq3_k1_tp_values = [mean(knn_lvq3_n2_1_tp),mean(knn_lvq3_n4_1_tp),mean(knn_lvq3_n6_1_tp),mean(knn_lvq3_n8_1_tp),mean(knn_lvq3_n10_1_tp)]
    lvq3_k3_tp_values = [mean(knn_lvq3_n2_3_tp),mean(knn_lvq3_n4_3_tp),mean(knn_lvq3_n6_3_tp),mean(knn_lvq3_n8_3_tp),mean(knn_lvq3_n10_3_tp)]
    lvq3_k1_fp_values = [mean(knn_lvq3_n2_1_fp),mean(knn_lvq3_n4_1_fp),mean(knn_lvq3_n6_1_fp),mean(knn_lvq3_n8_1_fp),mean(knn_lvq3_n10_1_fp)]
    lvq3_k3_fp_values = [mean(knn_lvq3_n2_3_fp),mean(knn_lvq3_n4_3_fp),mean(knn_lvq3_n6_3_fp),mean(knn_lvq3_n8_3_fp),mean(knn_lvq3_n10_3_fp)]

    print("K = 1: \ntp_rate_list_normal = [ {} ]\nfp_rate_list_normal = [ {} ] ".format(str(mean(knn_default_1_tp)),str(mean(knn_default_1_fp))))
    print("K = 1, LVQ1: \ntp_rate_list_lvq1 = {}\nfp_rate_list_lvq1 = {}".format(str(lvq1_k1_tp_values),str(lvq1_k1_fp_values)))
    print("K = 1, LVQ2.1: \ntp_rate_list_lvq2 = {}\nfp_rate_list_lvq2 = {}".format(str(lvq2_k1_tp_values),str(lvq2_k1_fp_values)))
    print("K = 1, LVQ3: \ntp_rate_list_lvq3 = {}\nfp_rate_list_lvq3 = {}".format(str(lvq3_k1_tp_values),str(lvq3_k1_fp_values)))
    print("K = 3: \ntp_rate_list_normal = [ {} ]\nfp_rate_list_normal = [ {} ] ".format(str(mean(knn_default_3_tp)),str(mean(knn_default_3_fp))))
    print("K = 3, LVQ1: \ntp_rate_list_lvq1 = {}\nfp_rate_list_lvq1 = {}".format(str(lvq1_k3_tp_values),str(lvq1_k3_fp_values)))
    print("K = 3, LVQ2.1: \ntp_rate_list_lvq2 = {}\nfp_rate_list_lvq2 = {}".format(str(lvq2_k3_tp_values),str(lvq2_k3_fp_values)))
    print("K = 3, LVQ3: \ntp_rate_list_lvq3 = {}\nfp_rate_list_lvq3 = {}".format(str(lvq3_k3_tp_values),str(lvq3_k3_fp_values)))

def pre_knn(df_train,df_test,k):

    test = df_test.reset_index(drop=True)

    if type(df_train) == type([]): 
        train = pd.DataFrame(df_train)
    else:
        train = df_train.reset_index(drop=True)
    
    test_labels = test['problems'].values
    test = test.drop(columns=['problems'])

    true_positive, false_positive, true_negative, false_negative= knn_normal( k, train, test, test_labels) # add predicted_values?
    tp_rate = true_positive/(true_positive + false_negative)
    fp_rate = false_positive/(false_positive + true_negative)
    #conf_matrix = confusion_matrix(test_labels, predicted_values)

    return true_positive, false_positive, true_negative, false_negative, tp_rate, fp_rate # conf_matrix
      
database_name_1 = 'kc1.arff'
database_name_2 = 'kc2.arff'

for i in [database_name_1]:
    data = arff.loadarff(i)

    df = pd.DataFrame(data[0])

    if i == 'kc2.arff':
        df['problems'] = df['problems'].apply(lambda x: x.decode("utf-8"))
        df['problems'] = df['problems'].map({"no": 0, "yes": 1})
        df['problems']
    elif i == 'kc1.arff':
        df.rename(columns = {'defects': 'problems'}, inplace = True)
        df['problems'] = df['problems'].apply(lambda x: x.decode("utf-8"))
        df['problems'] = df['problems'].map({"false": 0, "true": 1})
        df['problems']

    print("Start experiment for database {}".format(i))
    execute_experiment(df,[1,3],[2,4,6,8,10])
