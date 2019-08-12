# Opimization algorithms
# Author: Raphael Dang-Nhu
# 02/03/2019

import json
import collections
from scipy import signal
import itertools as it
import matplotlib.pyplot as pypl
import time
import numpy as np
import tensorflow as tf
import random
import logging
import datetime
import math
from project_conf import *
import math

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rl_tasks'))
from gym_project_conf import gym_fig_folder

# Function to setup the logger
def setup_logger(name, log_file, level=logging.INFO):
    
    formatter = logging.Formatter("%(message)s")
    fileH = logging.FileHandler(log_file,mode='w')
    fileH.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove previous handlers
    for h in logger.handlers:
        logger.removeHandler(h)

    logger.addHandler(fileH)
    
    return logger

# Generic class for algorithm
class Algorithm:
   
    # Init the algorithm with the configs
    def __init__(self,algo_config,mappings_config,opt_problem,debug_file,perf_file):
    
        # Get the config
        self.algo_config = algo_config
        self.mappings_config = mappings_config

        # Problem config
        self.nin = self.mappings_config['input']['n']
        self.non = self.mappings_config['output']['n']

        # Call computation graph
        self.create_graph()

        # Get the config 
        self.opt_prob = opt_problem
        self.function_config = self.opt_prob.function_config

        # Config for that problem
        self.debug_logger = setup_logger("debug",debug_file,logging.DEBUG)
        self.perf_logger = setup_logger("perf",perf_file,logging.INFO)
        
        return
    
    # Function to handle the bounds
    def handle_bounds(self):
    
        from multi_AS import MultiAS
            
        mapping_config = self.mappings_config
        opt_problem = self.opt_prob

        if(mapping_config["output"]["type"] == "throw ball"):
            self.output_bounds = mapping_config["output"]["bounds"]
        elif(mapping_config["output"]["type"] == "reach"):
            self.output_bounds = mapping_config["output"]["bounds"]
        elif("output bounds" in opt_problem.function_config):
            self.output_bounds = [opt_problem.function_config["output bounds"]]
        else:
            self.output_bounds = [comp["output bounds"] for comp in opt_problem.function_config["components"]]

        if(isinstance(self,MultiAS)):
            self.input_bounds = opt_problem.function_config["input bounds"][0]
            self.input_interval = (self.input_bounds[1] - self.input_bounds[0])/float(opt_problem.mapping_config["input"]["n"]-1)
            self.output_interval = (self.output_bounds[0][1] - self.output_bounds[0][1])/float(opt_problem.mapping_config["output"]["n"]-1)
            self.input_width = (self.input_bounds[1] - self.input_bounds[0])
            self.left_input_bounds = self.input_bounds[0]
        else:
            self.input_bounds = np.array(opt_problem.function_config["input bounds"])
            self.input_interval = (self.input_bounds[:,1] - self.input_bounds[:,0])/float(opt_problem.mapping_config["input"]["n"]-1)
            self.input_width = (self.input_bounds[:,1] - self.input_bounds[:,0])[:,np.newaxis]
            self.left_input_bounds = self.input_bounds[:,0][:,np.newaxis]


    def create_graph(self):
        # Default tensorflow graph for interval activation
        tf.reset_default_graph()
      
        # Variables for probabilities 
        self.probs = tf.get_variable("probas",dtype = tf.float32,initializer=tf.ones((self.nin,self.non))/2.)

        # Placeholder for i0
        self.i0 = tf.placeholder(tf.int32,shape=())
        self.jl = tf.placeholder(tf.int32,shape=())
        self.jr = tf.placeholder(tf.int32,shape=())

        # Variable part for the algorithms
        self.create_convolution()
        half_k = tf.floordiv(self.k_tf,2)

            # The range must be of size k exactly
        self.il = tf.math.maximum(tf.subtract(self.i0,half_k),tf.constant(0))
        self.ir = tf.math.minimum(tf.add(self.i0,tf.add(half_k,1)),tf.constant(self.nin-1))
        self.input_interval_size = self.ir - self.il

        # Placeholders for probability update

        # Sample random values for the synapses
        self.probs_slice = self.probs[self.il:self.ir] 
        self.samples = tf.random.uniform(shape=(self.input_interval_size,self.non))
        self.act_synapses = tf.cast(self.samples < self.probs_slice,tf.float32)
        self.act_pos = tf.ones(shape=(1,self.input_interval_size))
        self.activation_levels = tf.matmul(self.act_pos,self.act_synapses)


        # Create convolution
        self.input_tensor = tf.reshape(self.activation_levels,
                    [-1,1,self.non,1])
        self.interval_activation = \
                tf.reshape(tf.nn.conv2d(\
                input=self.input_tensor,
                filter=self.cnn_weights, strides=[1, 1, 1, 1],
                padding='SAME'),
                [self.non])

        # result is position of interval with highest activation
        self.argmax = tf.argmax(self.interval_activation)
        
        self.create_update()

        return

    # Activate with k. k is constant or taken from self.n_synapses[i0] 
    def activate(self):

            # Activate an input interval randomly
            i0 = np.random.randint(0,self.nin)

            #Debug line
            # Compute winner interval
            [il,ir,j0,k_log] = self.sess.run([self.il,self.ir,self.argmax,self.k_tf],feed_dict={self.i0:i0})[0:4]
            half_k = k_log //2
            jl = int(max(j0 - half_k,0))
            jr = int(min(j0 + half_k+1,self.non-1))
            return i0,il,ir,j0,jl,jr,k_log

    def log(self,message,level):
        if(level == logging.INFO):
            self.debug_logger.info(message)
            self.perf_logger.info(message)
        elif(level == logging.DEBUG):
            self.debug_logger.info(message)

    def log_problem_config(self):

        [[ilb,irb],[olb,orb]] = self.function_config["bounds"]


        self.log("PROBLEM CONFIGURATION",logging.INFO)
        self.log("Function name: "+self.function_config['name'],logging.INFO)

        # If config has field coeffs, log it
        if('coeffs' in self.function_config):
            self.log("Coeffs:",logging.INFO)
            self.log(self.function_config['coeffs'],logging.INFO)

        self.log("Number of input neurons: " + str(self.nin),logging.INFO)
        self.log("Number of output neurons: " + str(self.non),logging.INFO)
        self.log("Input left bound: " + str(ilb),logging.INFO)
        self.log("Input right bound: " + str(irb),logging.INFO)
        self.log("Output left bound: " + str(olb),logging.INFO)
        self.log("Output right bound: " + str(orb),logging.INFO)
        self.log("Expected output neuron: ",logging.DEBUG)
        self.log(self.opt_prob.expected_output_neuron,logging.DEBUG)

    
    def log_new_run(self,run):
        self.log("NEW RUN",logging.INFO)
        self.log("Run Number: "+str(run),logging.INFO)
        print("Run number "+str(run))

    # Functions to print the synapses
    def print_probs(self,probs,input_shape,save=False,episode=0):
       
        input_dim = len(input_shape)
        output_dim = probs.shape[input_dim]
        n_points = 4
        points = [(10*i,)*(input_dim-1) for i in range(n_points)]
        n_col = output_dim*n_points
        fig,axes = pypl.subplots(1,n_col,figsize=(15,5))
        axes_flat = axes.flat

        for dim in range(output_dim):
            for point in range(n_points):

                index = dim*n_points+point

                # Get dimension
                aux = [slice(None,None)]*input_dim+[slice(dim,dim+1)]
                sliced_probs = probs[aux]

                im = axes_flat[index].imshow(sliced_probs[points[point]][:,0,:],vmin=0.1,vmax=0.9)
                axes_flat[index].set_title("Point "+str(points[point]))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83,0.1,0.02,0.8])
        fig.colorbar(im,cax=cbar_ax)
        if(save):
            folder = gym_fig_folder+self.id+"/probs/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            pypl.savefig(folder+"episode_"+str(episode))
        else:
            pypl.show()
        pypl.close()
        #pypl.show(block=True)

    # Compute algorithm output (taken from multi_adaptive_k)
    def get_output(self,i0,k_input,k_output):


            half_k_output = k_output//2
            half_k_input = k_input//2
            slicing_bounds =\
                [tuple(\
                (max(i0[j]-half_k_input[i],0),
                min(i0[j]+half_k_input[i]+1,self.input_shape[0]))\
                        for j in range(len(i0))) for i in range(len(k_input))]
            slices = \
                [tuple(slice(x,y,1) for (x,y) in aux) for aux in slicing_bounds]
            slices_2 = [tuple(list(slices[i])+[slice(i,i+1)]) for i in range(len(slices))] 
            iterator =\
                [it.product(*[range(x,y) for (x,y) in aux]) for aux in slicing_bounds]

            # Multiply with tensor full of ones
            slices_shape = [tuple(
                    y-x for (x,y) in aux) for aux in slicing_bounds] 
            ones = [np.ones(aux) for aux in slices_shape] 
            sliced_probs = [self.probs[slices_2[i]] for i in range(len(slices))] 
            act_synapses = [np.random.random_sample(aux.shape) < aux for aux in sliced_probs]
            activation = [np.tensordot(ones[i],act_synapses[i],len(slices_shape[i]))[0] for i in range(len(slices_shape))]

            
            # Do the convolution on output
            n_dims = len(activation[0].shape)
            convolution_shape = [(aux,)*n_dims for aux in k_output]
            kernel = [np.ones(aux) for aux in convolution_shape]
            convolved_activation = np.array([ 
                    signal.fftconvolve(activation[i],kernel[i],mode='same')\
                            for i in range(len(activation)) ])

            # Get argmax for maximum activation
            j0_choice = [np.argwhere(conv_act == np.amax(conv_act)) for conv_act in convolved_activation]
            choice = [np.random.choice(len(aux)) for aux in j0_choice]
            j0 = [j0_choice[i][choice[i]] for i in range(len(j0_choice))]
            
            return j0,slicing_bounds,slices,iterator


    
    # Get iterator from list of bounds 
    def get_iterator(self,bounds):
        return it.product(*[range(x,y) for (x,y) in bounds])

    def get_slices(self,bounds):
        return tuple([slice(x,y) for (x,y) in bounds])

    # Function to update all synapses corresponding to list of activations
    def iterate_fixed_k(self,function,half_k):

        for (ih,jh) in self.list_activations:

                # Slicing bounds for input
                slicing_bounds =\
                    tuple(\
                          (max(ih[i]-half_k,0),
                           min(ih[i]+half_k+1,self.input_shape[0]))\
                          for i in range(len(ih)))
        
                iterator = self.get_iterator(slicing_bounds)

                # Slicing bounds for the output
                output_slicing_bounds =\
                    [tuple(
                    (max(aux[i]-half_k,0),
                    min(aux[i]+half_k+1,self.output_shape[1]))\
                          for i in range(len(aux)) ) for aux in jh]
                
                # For each input neuron
                for input_neuron in iterator:

                    output_iterators = [self.get_iterator(bounds) for bounds in output_slicing_bounds]
                    
                    # For each output dimension
                    for i in range(len(output_iterators)):

                        out_iterator = output_iterators[i]

                        # For each output neuron
                        for output_neuron in out_iterator:
                            

                            self.probs[input_neuron][i,output_neuron] = function(self.probs[input_neuron][i,output_neuron])
        return

    def get_output_value(self,j0):
        # If throw ball
        conf = self.mappings_config["output"]
        type_map = conf["type"] 
        if(type_map == "throw ball"):
            if("grid" in conf and conf["grid"] == "true"):
                value = [self.opt_prob.output_neuron_value[0][j0[0][0]][j0[0][1]]]
            else:
                value = [self.opt_prob.output_neuron_value[0][j0[0][0]][j0[1][0]]]
        elif(type_map == "reach"):
            if(("grid" in conf and conf["grid"] == "true") or self.algo_config["name"] == "reinforce"):
                value = self.opt_prob.output_neuron_value[0][j0[0][0]][j0[0][1]]
            else:
                print("j0",j0)
                value = self.opt_prob.output_neuron_value[0][j0[0][0]][j0[1][0]]
            #print("value",value)
        else:
            if(not np.isscalar(self.opt_prob.output_neuron_value[0][0])):
                value = [ self.opt_prob.output_neuron_value[0][i][j0[i]] for i in range(self.opt_prob.output_mapping.dim) ]
            else:
                value = [ self.opt_prob.output_neuron_value[i][j0[i]] for i in range(self.opt_prob.output_mapping.dim) ]
        return value

    # Function to compute the value from j0
    def compute_error(self,i0,j0):

        value = self.get_output_value(j0)

        ############################################
        ##### Added to ensure compatibility
        ##### with multi-dimensional output functions
        ############################################
        #############################################
        ####### Changed to depend on feedback #######
        exp_value = self.opt_prob.expected_output_value[i0]
        error = self.opt_prob.feedback.get_error(value,exp_value)
        return value,exp_value,error

    # Functions to save synaps figures the synapses
    def save_print_probs(self,probs,input_shape,file_name):
        
        output_dim = probs.shape[len(input_shape)]
        for dim in range(output_dim):
            pypl.subplot(1,output_dim+1,dim+1)
            pypl.imshow(probs[:,dim,:])
            #pypl.colorbar()
        pypl.savefig(file_name)
        pypl.close()
        #pypl.show(block=True)
    
    ########################################################
    ######## Functions to evaluate the model ###############
    ########################################################

    # Function to sample data
    def sample_data(self,n_samples,split=0):
        
       
        if(split==0):
            data = (np.random.rand(n_samples,self.input_dim,1)*self.input_width + self.left_input_bounds)[:,:,0]
        elif(split==1):
            data = (np.random.rand(n_samples,self.input_dim,1)*self.input_width/2. + self.left_input_bounds)[:,:,0]
        elif(split==2):
            data = (np.random.rand(n_samples,self.input_dim,1)*self.input_width/2 + self.left_input_bounds + self.input_width/2)[:,:,0]
        labels = np.array([[self.opt_prob.function(data[i])] for i in range(n_samples)])
        # Compatibility for case reach
        if(len(labels.shape)>=3):
            labels = labels[:,0]

        # Get indexes
        index_data = [((data[i]+self.input_interval/2.-self.left_input_bounds)//self.input_interval).astype(int) for i in range(n_samples)]
        return data,labels,index_data

    
    # Function to get k form number of synapses
    def get_k_input_output(self,i0):
        
        if(hasattr(self,'static') and self.static):
            k_input = np.full(self.n_outputs,self.k_static)
            k_output = np.full(self.n_outputs,self.k_static)
            return k_input,k_output

        aux = self.opt_prob.output_mapping.config
        if("grid" in aux and aux["grid"] == "true"):
            k_output = np.array([math.ceil(math.sqrt(max(aux,1)) / self.k_output_factor) for aux in self.n_synapses[i0]] )
            k_input = np.array([math.ceil(math.sqrt(max(aux,1)) / self.k_input_factor) for aux in self.n_synapses[i0]] )
        else:

            type_mapp = self.opt_prob.mapping_config["output"]["type"]
            if(type_mapp == "reach" or type_mapp == "throw ball"):
                    k_input = np.array([max(aux // self.k_input_factor,1) for aux in self.n_synapses[i0]] )
                    k_output = np.array([max(aux // self.k_output_factor,1) for aux in self.n_synapses[i0]] )
            else: 
                    k_output = np.array([math.ceil(max(aux // self.k_output_factor,1)) for aux in self.n_synapses[i0]] )
                    k_input = np.array([math.ceil(max(aux // self.k_input_factor,1)) for aux in self.n_synapses[i0]] )
        return k_input,k_output

    def evaluate(self,n,split=0,step=0):
        from multi_AS import MultiAS
        
        if(split == 0):
            data,label,index_data = self.sample_data(n)
        elif(split == 1):
            data,label,index_data = self.sample_data(n,1)
        elif(split==2):
            data,label,index_data = self.sample_data(n,2)
        
        mean = 0.
        list_preds = []
        for i in range(n):
            if(isinstance(self,MultiAS)):
                j0 = self.compute_output_activation(index_data[i])
            else:
                    i0 = tuple(index_data[i][0])
                    k_input,k_output = self.get_k_input_output(i0)
                    j0,_,_,_ = self.get_output(index_data[i][0],k_input,k_output)
            aux = self.get_output_value(j0) 
            # Onlt work for L1 feedback
            delta = self.opt_prob.feedback.get_error(label[i],aux)
            mean += delta
            list_preds.append(aux)
        
        if(False):
            pypl.figure()
            x = [aux[0] for aux in list_preds]
            y = [aux[1] for aux in list_preds]
            pypl.scatter(x,y)
            pypl.axvline(x=0.354)
            pypl.axvline(x=0.707)
            pypl.axhline(y=0.354)
            pypl.axhline(y=0.707)
            pypl.savefig(self.scatter_folder+str(step))
            pypl.close()
        return mean/float(n)

    # Functions to log batch error
    def log_batch_error(self,step,incr=False):
        if((step % self.step_test == 0) & (self.n_test > 0)):
            if(incr):
                batch_error = self.evaluate(self.n_test,1)
                self.log("Step: " + str(step),logging.INFO)
                self.log("Error1: " + str(batch_error),logging.INFO)
                batch_error = self.evaluate(self.n_test,2)
                self.log("Step: " + str(step),logging.INFO)
                self.log("Error2: " + str(batch_error),logging.INFO)
                batch_error = self.evaluate(self.n_test,0,step)
                self.log("Step: " + str(step),logging.INFO)
                self.log("Error: " + str(batch_error),logging.INFO)
                    
            else:
                batch_error = self.evaluate(self.n_test,0,step)
                self.log("Step: " + str(step),logging.INFO)
                self.log("Error: " + str(batch_error),logging.INFO)

    def log_test_cross_error(self):
        # Log test and cross error
        if self.n_test > 0:
            test_error = self.evaluate(self.n_test)
            self.log("Test Error: " + str(test_error),logging.INFO)
        if self.n_cross > 0:
            cross_error = self.evaluate(self.n_cross)
            self.log("Cross Error: " + str(cross_error),logging.INFO)

    # Read step intervals
    def read_n_steps(self,common_config):

        self.M = common_config['M']
        self.number_of_runs = common_config['runs']

        if("n test" in common_config):
            self.n_test = common_config["n test"]
        else:
            self.n_test = 0
        if("step test" in common_config):
            self.step_test = common_config["step test"]
        else:
            self.step_test = 1000000
        if("n cross" in common_config):
            self.n_cross = common_config["n cross"]
        else:
            self.n_cross = 0

    # Read dims
    def read_dim(self):
        self.input_dim = self.opt_prob.function.input_dim
        self.output_dim = self.opt_prob.output_mapping.dim

    # Chedk incremental
    def is_incremental(self,algo_config):
        if("incremental" in algo_config and algo_config["incremental"] == 1):
            self.incremental = True
            self.half_M = self.M//2
        else:
            self.incremental = False

    def is_static(self,algo_config):
        # Static mode
        if("static" in algo_config):
            self.static = True
            self.k_static = algo_config["static"]
        else:
            self.static = False

    # Compute the initial average error for neurons
    def compute_init_average_error(self):
        

        # Else
        if("components" in self.function_config):
            res = 0
            for comp in self.function_config["components"]:
                [lb,rb] = comp["output bounds"]
                res += rb - lb
            return res/3.

        # If throw ball
        elif(self.mappings_config["output"]["type"] == "throw ball"):
            [lb,rb] = self.function_config["output bounds"]
            return (rb-lb)/3.
        else:
            [lb,rb] = self.function_config["output bounds"]
            return (rb-lb)/3.

    # Create logger
    def create_logger(self,debug_file,perf_file):
        # Log config 
        self.debug_logger = setup_logger("debug",debug_file,logging.DEBUG)
        self.perf_logger = setup_logger("perf",perf_file,logging.INFO)
        
        # Log parameters
        self.log("Algorithm",logging.INFO)
        self.log(json.dumps(self.algo_config,indent=4),logging.INFO)
        self.log("Function",logging.INFO)
        self.log(json.dumps(self.function_config,indent=4),logging.INFO)
