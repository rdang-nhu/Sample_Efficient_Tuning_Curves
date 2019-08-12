# Benchmark implementation of reinforce algorithm for supervised learning
# Author: Raphael Dang-Nhu
# 12/06/2019

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

class Reinforce(algorithms.Algorithm):

    # Create the Keras 
    def create_model(self,algo_config):

        #tf.reset_default_graph()
        self.coeff = algo_config["coeff"]
        self.alpha = algo_config["alpha error"]

        input_layer = keras.Input(shape=(self.input_dim,))
        width = algo_config['width layers']

        # Define all layers
        self.layer1 = Dense(width,activation='relu')(input_layer) 
        self.output = Dense(self.output_size,activation='softmax')(self.layer1)

        self.model = keras.Model(inputs=input_layer,outputs=self.output)
        # Get the weights (we use a custom optimization procedure here)
        self.model_weights = self.model.trainable_weights


        # Get the gradients
        output_one_hot = keras.Input(shape=(1,),dtype=tf.int32)
        grads = keras.backend.gradients(\
                # Custom loss
                self.output[0][output_one_hot[0][0]],
                # Variables
                self.model_weights)

        self.function1 = keras.backend.function([input_layer,output_one_hot],grads)
        self.function2 = keras.backend.function([input_layer,output_one_hot],[self.output[0][output_one_hot[0][0]]])
    
        return

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
        

        # Input dim
        self.input_dim = opt_problem.input_mapping.dim
        self.output_dim = opt_problem.output_mapping.dim
        self.output_width = opt_problem.output_mapping.config["n"]
        self.output_size = self.output_width**self.output_dim
        
        # Get the interval
        self.handle_bounds()

        # Log config 
        self.create_logger(debug_file,perf_file)
        
        return

    def convert_j0(self,j0):
            test = self.mappings_config["output"]["type"] 
            if(test == "throw ball" or test == "reach0"):
                res = [[j0[0] %  self.output_width, j0[0] // self.output_width]]
                return res
            elif(test == "reach"):
                res = [[j0[0] %  self.output_width, j0[0] // self.output_width]]
                return res
            return [j0]
    #Compute init error
    
    def predict(self,data):
            probs = self.model.predict(np.array([data,]))
            output_one_hot = np.random.choice(len(probs[0]),1,p=probs[0])
            j0 = self.convert_j0(output_one_hot)
            return j0,[output_one_hot]

    def compute_error(self,n):
            data,labels,_ = self.sample_data(n)
            error = 0.
            for i in range(n):
                # Evaluate
                j0,output_one_hot = self.predict(data[i])

                # Convert to value
                value = self.get_output_value(j0)

                # Compare with label
                E = self.opt_prob.feedback.get_error(value,labels[i])
                error += float(E)/n
            return error

    # Function to log batch error
    def log_batch_error(self,step):
        if(step % self.step_test == 0):
            n_test = self.n_test
            error = self.compute_error(n_test)
            self.log("Step:" + str(step),logging.INFO)
            self.log("Error: " + str(error),logging.INFO)

    # Function to log batch error
    def log_test_error(self):
        error = self.compute_error(self.n_test)
        error1 = self.compute_error(self.n_cross)
        self.log("Test Error: " + str(error),logging.INFO)
        self.log("Cross Error: " + str(error1),logging.INFO)

    # Main function
    def solve(self):
        
        for run in range(self.number_of_runs):

            # Create model
            self.create_model(self.algo_config)
            av_error = self.compute_error(100)

            self.log_new_run(run)
            
            # Sample the training data
            data,labels,_ = self.sample_data(self.M)
           
            for step in range(0,self.M):

                # TEST
                self.log_batch_error(step)

                # Train the neural network
                # Evaluate
                j0,output_one_hot = self.predict(data[step])

                # Convert to value
                value = self.get_output_value(j0)


                # Compare with label
                E = self.opt_prob.feedback.get_error(value,labels[step])
                # Update weights with reinforce rule
                grads = self.function1( [[data[step]],output_one_hot])
                probas = self.function2([[data[step]],output_one_hot])


                # Multiply by delta/proba
                scaled_grads = self.coeff*(av_error-E)*np.array(grads)/max(probas[0],0.0000001)
                #scaled_grads = self.coeff*(-E)*np.array(grads)/max(probas[0],0.0000001)

                # Update weights                
                weights = np.array(self.model.get_weights())
                self.model.set_weights(weights+scaled_grads)

            self.log_test_error()

        return 
