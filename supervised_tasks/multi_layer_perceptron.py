# Benchmark implementation using a neural network
# Author: Raphael Dang-Nhu
# 09/05/2019

import time
import numpy as np
import tensorflow as tf
import random
import logging
import datetime
import math
import algorithms
from project_conf import *
import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.layers import Dense

# Modified to take sevral bounds into account
class custom_activation:

    def __init__(self,i,bounds):
        self.left = bounds[i][0]
        self.right = bounds[i][1]

    def __call__(self,x):
        res =  keras.backend.sigmoid(x) * (self.right - self.left)+self.left
        return res

class Perceptron(algorithms.Algorithm):

    
    # Create the Keras model
    def create_model(self,algo_config):
        
        #Incremental
        self.is_incremental(algo_config)

        #tf.reset_default_graph()
        self.batch_size = algo_config['batch size']

        self.inputs = keras.Input(shape=(self.input_dim,))
        width = algo_config['width layers']

        # Define all layers
        self.layer1 = [ Dense(width,activation='relu')(self.inputs) for _ in range(self.output_dim)]
        self.layer2 = [ Dense(width,activation='relu')(self.layer1[i]) for i in range(self.output_dim)]

        # Check if throw_ball
        if(self.mappings_config["output"]["type"] == "throw ball"):
            self.acts = [custom_activation(i,self.output_bounds) for i in range(self.output_dim)]
            self.output = [ Dense(1)(self.layer2[i]) for i in range(self.output_dim) ]
            # Sin(2 .) the first output
            aux0 = keras.layers.Lambda(lambda x : 2*x)(self.output[0])
            aux1 = keras.layers.Activation(tf.math.sin)(aux0)
            # Square the second output 
            aux2 = keras.layers.Activation(tf.math.square)(self.output[1])
            # Multiply both ouputs
            self.merged_output = keras.layers.Multiply()([aux1,aux2])
            self.model = keras.Model(inputs=self.inputs,outputs=self.merged_output)
            self.debug_model = keras.Model(inputs=self.inputs,outputs=[self.merged_output,aux1,aux2])
        elif(self.mappings_config["output"]["type"] == "reach"):
            self.acts = [custom_activation(i,self.output_bounds) for i in range(self.output_dim)]
            self.output = [ Dense(1)(self.layer2[i]) for i in range(self.output_dim) ]

            # Get paramters
            d1 = self.mappings_config["output"]["length long arm"]
            d2 = self.mappings_config["output"]["length short arm"]
            # Get cos and sins
            cos_alpha = keras.layers.Activation(tf.math.cos)(self.output[0])
            sin_alpha = keras.layers.Activation(tf.math.sin)(self.output[0])
            alpha_beta = keras.layers.Add()([self.output[0],self.output[1]])
            cos_alpha_beta = keras.layers.Activation(tf.math.cos)(alpha_beta)
            sin_alpha_beta = keras.layers.Activation(tf.math.sin)(alpha_beta)
            # Multiply cos and sin
            aux1 = keras.layers.Lambda(lambda x : d1*x[0] + d2*x[1])([cos_alpha,cos_alpha_beta])
            aux2 = keras.layers.Lambda(lambda x : d1*x[0] + d2*x[1])([sin_alpha,sin_alpha_beta])
            # Multiply both ouputs
            outputs = keras.layers.Concatenate()([aux1,aux2])
            self.model = keras.Model(inputs=self.inputs,outputs=outputs)
            self.debug_model = keras.Model(inputs=self.inputs,outputs=[outputs,cos_alpha,sin_alpha,cos_alpha_beta,sin_alpha_beta])
        else:
            self.acts = [custom_activation(i,self.output_bounds) for i in range(self.output_dim)]
            self.output = [ Dense(1,activation=self.acts[i])(self.layer2[i]) for i in range(self.output_dim) ]
            self.model = keras.Model(inputs=self.inputs,outputs=self.output)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(algo_config['learning rate'])

        self.model.compile(optimizer,loss='mse',metrics=['mean_absolute_error'])



    def __init__(self,algo_config,common_config,mapping_config,opt_problem,debug_file,perf_file):

        # Get the config
        self.algo_config = algo_config
        self.mappings_config = mapping_config
        self.opt_prob = opt_problem
        self.function_config = self.opt_prob.function_config
        
        # Initial config of metaparameters
        # Number of steps
        self.read_n_steps(common_config)

        # Number of runs
        self.number_of_runs = common_config['runs']
        self.epochs = algo_config['epochs']

        # Input dim
        self.input_dim = opt_problem.function_config['input dimension']
        self.output_dim = opt_problem.output_mapping.dim
        
        # Get the interval
        self.handle_bounds()

        # Log config 
        self.create_logger(debug_file,perf_file)
        
        # Log parameters
        self.log("Output values:",logging.DEBUG)
        self.log(self.opt_prob.output_neuron_value,logging.DEBUG)
        self.log("\n",logging.DEBUG)

        return
    

    # Main function
    def solve(self):
        
        for run in range(self.number_of_runs):

            # Create model
            self.create_model(self.algo_config)
            
            self.log_new_run(run)
            
            # Sample the training data
            if(self.incremental):
                data,labels,_ = self.sample_data(self.half_M,1)
                data2,labels2,_ = self.sample_data(self.M-self.half_M,2)
                data=np.append(data,data2)
                labels=np.append(labels,labels2)
            else:
                data,labels,_ = self.sample_data(self.M)
           
            for step in range(0,self.M,self.batch_size):

                # TEST
                if(step % self.step_test == 0):
                    if(self.incremental):
                        test_data,test_labels,_ = self.sample_data(self.n_test,1)
                        batch_error = self.model.evaluate(test_data,test_labels)
                        self.log("Step: " + str(step),logging.INFO)
                        self.log("Error1: " + str(batch_error[1]),logging.INFO)
                        test_data,test_labels,_ = self.sample_data(self.n_test,2)
                        batch_error = self.model.evaluate(test_data,test_labels)
                        self.log("Step:" + str(step),logging.INFO)
                        self.log("Error2: " + str(batch_error[1]),logging.INFO)
                            
                    else:
                        
                        test_data,test_labels,_ = self.sample_data(self.n_test)
                        loss = self.model.evaluate(test_data,test_labels)
                        
                        # Log
                        self.log("Step: " + str(step),logging.INFO)
                        self.log("Error: " + str(loss[1]),logging.INFO)

                # Train the neural network
                slice_ = slice(step,min(step+self.batch_size,self.M))
                self.model.fit(data[slice_],labels[slice_],epochs=self.epochs,batch_size=self.batch_size)


            # CROSS VALIDATION
            cross_data,cross_labels,_ = self.sample_data(self.n_cross)
            cross_loss = self.model.evaluate(cross_data,cross_labels)


            # TEST
            test_data,test_labels,_ = self.sample_data(self.n_test)
            loss = self.model.evaluate(test_data,test_labels)

            
            self.log("Test Error: " + str(loss[1]),logging.INFO)
            self.log("Cross Error: " + str(cross_loss[1]),logging.INFO)

        return 
