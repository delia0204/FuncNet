# time 2018.12
# author lmx

import tensorflow as tf
import numpy as np
import os
import json
import sys
 
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        self._saver = None
        self._centroid_grid = None
        self._trained = False
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
 
            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]), dtype=tf.int64)), [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects, tf.stack( [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            print iter_no
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        if self._centroid_grid is None:
            #Store a centroid grid for easy retrieval later on
            centroid_grid = [[] for i in range(self._m)]
            self._weightages = list(self._sess.run(self._weightage_vects))
            self._locations = list(self._sess.run(self._location_vects))
            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weightages[i])
            self._centroid_grid = centroid_grid
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect - self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return

    def load(self, model_path):
        if self._saver is None:
            with self._graph.as_default():
                self._saver = tf.train.Saver()
        if model_path is not None:
            self._saver.restore(self._sess, model_path)
            self._trained = True
            print("Model loaded.")
        else:
            print("Model path dosen't specified.")

    def save(self, save_path):
        if self._saver is None:
            with self._graph.as_default():
                self._saver = tf.train.Saver()
        if save_path is not None:
            self._saver.save(self._sess, save_path)
            print("model saved.")

def som_map(dim_x, dim_y, valid_file, map_file):
    test_data = []
    test_label = []
    test_data_nor = []
    #test auc
    som = SOM(dim_x, dim_y, 64, 100)
    som.load("./saved_model/mapping_model")
    #test data
    valid_line = 0.0
    with open(valid_file, 'r') as inf:
        for line in inf:
            valid_line += 1
            f_info = json.loads(line.strip())
            test_data.append(f_info['vector'])
            test_label.append(f_info['fname'])
            test_data_nor.append(f_info['vector_nor'])

    np_test_data = np.array(test_data)
    #map
    som.get_centroids()
    mapped = som.map_vects(np_test_data)
    #store map data
    for i,m in enumerate(mapped):
        index = m[1]*(dim_x)+m[0]
        with open(map_file, 'a') as outf:
            recoder = {'fname':test_label[i], 'vector_nor':test_data_nor[i], 'pos':index}
            json.dump(recoder, outf, ensure_ascii=False)
            outf.write('\n')

def save_model_train(dim_x,dim_y,train_file):
    train_data = []
    train_label = []
    #input
    with open(train_file, 'r') as inf:
        for line in inf:
            f_info = json.loads(line.strip())
            train_data.append(f_info['vector'])
            train_label.append(f_info['fname'])

    np_train_data = np.array(train_data)
    
    #Train a 20x20 SOM with 400 iterations
    som = SOM(dim_x, dim_y, 64, 100)

    #train
    som.train(np_train_data)
    
    #save model
    som.save("./saved_model/mapping_model")



if __name__ == "__main__":
    #init
    auc = 0.0
    dim_x = 100
    dim_y = 100


    # train no norm: valid - 0.62
    train_file = 'a0_norm.json'
    valid_file = sys.argv[1]
    map_file = sys.argv[2]
    
    #save_model
    #save_model_train(dim_x,dim_y,train_file)

    #valid model
    som_map(dim_x,dim_y, valid_file, map_file)