# time 2018.12
# author lmx

import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json
import sys
import datetime
#usage: python .py target.json src.json result 

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')
parser.add_argument('--fname', type=str, default='./data/sh.json',
        help='epoch number')
parser.add_argument('--rname', type=str, default='./data/sh_vec.json', 
        help='batch size') 
parser.add_argument('--load_path', type=str,
        default='./saved_model/graphnn-model_best',
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def main():
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path

    F_NAME = args.fname
    R_NAME = args.rname

    FUNC_NAME_LIST = []
    FUNC_NAME_LIST = get_func_list(F_NAME)
    FUNC_NAME_LIST_LEN = len(FUNC_NAME_LIST)

    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    with open(F_NAME) as inf:
        label = 0
        for line in inf:
            g_info = json.loads(line.strip())

            #init graph
            cur_graph = graph(g_info['n_num'], label, g_info['src'])
            for u in range(g_info['n_num']):
                cur_graph.features[u] = np.array(g_info['features'][u])
                for v in g_info['succs'][u]:
                    cur_graph.add_edge(u, v)

            #feature data -> matrix
            feature_dim = 7
            X_input = np.zeros((1,cur_graph.node_num, feature_dim))
            node_mask = np.zeros((1,cur_graph.node_num, cur_graph.node_num))
            for u in range(cur_graph.node_num):
                X_input[0,u, :] = np.array( cur_graph.features[u] )
            for v in cur_graph.succs[u]:
                node_mask[0,u, v] = 1

            #write into R_NAME
            with open( R_NAME, 'a') as rf:
                graph_id = g_info['fname']
                graph_vec, graph_vec_nor = gnn.get_embed(X_input, node_mask)

                #overwrite encoder-> myencoder: http://www.bubuko.com/infodetail-2757267.html
                recoder = {"fname":graph_id,"vector":graph_vec[0],"vector_nor":graph_vec_nor[0]}
                json.dump(recoder, rf, cls=MyEncoder)
                rf.write('\n')

            label+=1


if __name__ == '__main__':
    main()
