# time 2018.11.11  24:00
# author lmx

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from graphnnSiamese import graphnn
import json

#joint file_name
#return file_name_list
def get_f_name(DATA, SF, CM, OP, VS):
    F_NAME = []
    for sf in SF:
        for cm in CM:
            for op in OP:
                for vs in VS:
                    F_NAME.append(DATA+sf+cm+op+vs+".json")
    return F_NAME

#return function_name_list 
def get_func_list(F_NAME):
    name_num = 0
    name_list = []

    with open(F_NAME) as inf:
        for line in inf:
            g_info = json.loads(line.strip())
            name_list.append(g_info['fname'])
    return name_list

#give index, order function_name 
#return dict[function_name]=index
def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    return name_dict

class graph(object):
    def __init__(self, node_num = 0, label = None, name = None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.features = []
        self.succs = []
        self.preds = []
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                
    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret

#recoder graph_of_func, class_label
#return graph_list, func_index_list
def read_all_graphs(F_NAME, FUNC_NAME_DICT, FEATURE_DIM):
    graphs = []
    classes = []
    #create list[] for each func
    if FUNC_NAME_DICT != None:
        for f in range(len(FUNC_NAME_DICT)):
            classes.append([])

    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                label = FUNC_NAME_DICT[g_info['fname']]
                classes[label].append(len(graphs))
                cur_graph = graph(g_info['n_num'], label, g_info['src'])
                for u in range(g_info['n_num']):
                    cur_graph.features[u] = np.array(g_info['features'][u])
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)
    return graphs, classes

#[([(G00,G10)],[]),([(G00,G11)],[]),...]
def generate_epoch_pair(Gs, classes, M, FUNC_NAME_DICT1_LEN, FUNC_NAME_DICT2_LEN, output_id = False, load_id = None):
    epoch_data = []
    id_data = [] 

    #all match pair
    for s1 in range(FUNC_NAME_DICT1_LEN):
        for s2 in range(FUNC_NAME_DICT2_LEN):
            X1, X2, m1, m2, y, pos_id, neg_id = get_pair(Gs, classes,M, st1=s1, st2=(FUNC_NAME_DICT1_LEN+s2), output_id=True)
            id_data.append((pos_id, neg_id))
            epoch_data.append( (X1,X2,m1,m2,y))

    return epoch_data, id_data
   

#generate ( [(st1,st2)],[] )
def get_pair(Gs, classes, M, st1 = -1, st2 = -1, output_id = False, load_id = None):

    pos_ids = []
    neg_ids = []
    pos_ids.append((st1,st2))
        
    M_pos = 1
    M_neg = 0
    M = 1

    maxN1 = 0
    maxN2 = 0
    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)

    feature_dim = len(Gs[0].features[0])
    X1_input = np.zeros((M, maxN1, feature_dim))
    X2_input = np.zeros((M, maxN2, feature_dim))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    y_input = np.zeros((M))
    
    for i in range(M_pos):
        y_input[i] = 1
        g1 = Gs[pos_ids[i][0]]
        g2 = Gs[pos_ids[i][1]]
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1

    return X1_input,X2_input,node1_mask,node2_mask,y_input,pos_ids,neg_ids


def train_epoch(model, graphs, classes, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))   #Random shuffle

    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X1, X2, mask1, mask2, y = cur_data
        loss = model.train(X1, X2, mask1, mask2, y)
        cum_loss += loss

    return cum_loss / len(perm)

def get_relation_matrix(model, graphs, classes, batch_size, FUNC_NAME_DICT1_LEN, FUNC_NAME_DICT2_LEN,load_data=None):
    tot_diff = []
    relation_matrix = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    #get all diffs
    m = []
    for cur_data in epoch_data:
        X1, X2, m1, m2,y  = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)
        #print diff
        tot_diff += list(diff)
        m.append(diff[0])
    #print len(m)
    #print m

    #into matrix
    for st1 in range(FUNC_NAME_DICT1_LEN):
        m1 = []
        for st2 in range(FUNC_NAME_DICT2_LEN):
            m1.append(m[(st1*FUNC_NAME_DICT2_LEN + st2)])
        relation_matrix.append(m1)
    return relation_matrix


def get_auc_epoch(model, graphs, classes, batch_size, load_data=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data


    for cur_data in epoch_data:
        X1, X2, m1, m2,y  = cur_data
        diff = model.calc_diff(X1, X2, m1, m2)
        #print diff

        tot_diff += list(diff)
        tot_truth += list(y > 0)


    diff = np.array(tot_diff)
    truth = np.array(tot_truth)

    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    model_auc = auc(fpr, tpr)

    return model_auc, fpr, tpr, thres
