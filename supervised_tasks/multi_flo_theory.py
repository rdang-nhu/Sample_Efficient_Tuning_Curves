# Adaptive K algorithm with multi dimensional mappings
# Author: Raphael Dang-Nhu
# 15/04/2019

import json
import pprint
import os
import functools
from operator import mul
import itertools as it
from scipy import signal
import time
import numpy as np
import tensorflow as tf
import random
import logging
import datetime
import math
import algorithms
from project_conf import *
import matplotlib.pyplot as pypl

class MultiFloTheory(algorithms.Algorithm):

    def __init__(self,algo_config,common_config,mapping_config,opt_problem,debug_file,perf_file):

        self.name=algo_config['name']
        # Initial config of metaparameters
        # Number of steps
        self.read_n_steps(common_config)
        # Alpha parameter for exp decaying mean error
        self.alpha = algo_config['alpha']

        
        # Incremental mode
        self.is_incremental(algo_config)
        self.is_static(algo_config)
        if(hasattr(self,'static') and self.static):
            self.k_static=algo_config['static']

        # Average mode for synapses
        if(algo_config['mode'] == "exponential"):
            self.mode = 0
            self.alpha_synapse = algo_config['alpha synapse']
        elif(algo_config['mode'] == "average"):
            self.mode = 1
        else:
            raise ValueError("Average mode has unvalid value")
        
        #Plot synapses
        if("plot synapses" in algo_config and algo_config["plot synapses"]==1):
            self.plot_synapses = True

            self.steps_plot = algo_config["steps plot"] 
        else:
            self.plot_synapses = False

        # Number of runs
        self.number_of_runs = common_config['runs']
        # Parameters for security margin
        if(not "error coeff" in algo_config):
            algo_config["error_coeff"] = 1
        print("error coeff",algo_config["error coeff"])
        self.error_coeff = algo_config["error coeff"]
        # Synapse counter threshold
        self.counter_threshold = algo_config['counter threshold']
        self.counter_neuron_threshold = algo_config['counter neuron threshold']
        self.prune_constant = algo_config['prune_constant']
        self.k_output_factor = algo_config['k output factor'] # k_output is num_syn/k_output_factor
        self.k_input_factor = algo_config['k input factor']# k input is num_syn/k_input_factor
        #remark: k_input_factor should be small if output curve to learn is flat and large otherwise (output curve means hear the values encoded in the output layers)
        # if the  derivative of the output curve is at most b, then k_input_factor = b  
        if(hasattr(self,'static') and self.static):
            self.min_number_synapses_factor= algo_config['min_num_synapse_factor']
            self.minimal_number_synapses = self.k_static * self.min_number_synapses_factor
        else :              
            self.minimal_number_synapses =  algo_config['minimal_number_synapses']
        
        # Get the config
        self.algo_config = algo_config
        self.mappings_config = mapping_config
        self.opt_prob = opt_problem
        self.function_config = self.opt_prob.function_config
        
        # Get the intervals
        self.handle_bounds()

        self.read_dim() 
        
        # Check if grid for output
        aux = self.opt_prob.output_mapping.config
        if("grid" in aux and aux["grid"] == "true"):
            self.grid = True
        else:
            self.grid = False

        # Log config 
        self.debug_logger = algorithms.setup_logger("debug",debug_file,logging.DEBUG)
        self.perf_logger = algorithms.setup_logger("perf",perf_file,logging.INFO)
        self.scatter_folder = fig_folder+"scatter/"+str(int(time.time()))+"/"
       
        if(not os.path.exists(self.scatter_folder)):
            os.makedirs(self.scatter_folder)
        
        # Log parameters
        self.log(json.dumps(self.algo_config,indent=4),logging.INFO)
        return

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

    def plot_synapse_history(self):
       
        #pypl.style.use("classic")
        pypl.title("Evolution of Synapses Weight, Second-Order Polynomial")
        pypl.xlabel("Input neuron")
        pypl.ylabel("Output neuron")
        fig, axes = pypl.subplots(nrows=3,ncols=5,sharex=True,sharey=True,figsize=[10,6])
        for i in range(len(axes.flat)):
                print(i)
                ax = axes.flat[i]
                im = ax.imshow(self.record_synapses[i].reshape(100,100).T, vmin=0, vmax=1,cmap='plasma')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                ax.set_title("Step "+str(self.steps_plot[i]),size=7)

        #pypl.show()
        #pypl.suptitle("Synapses Weights, Second-Order Polynomial")
        fig.text(0.5, 0.04, 'Input neuron', ha='center')
        fig.text(0.04, 0.5, 'Output neuron', va='center', rotation='vertical')
        pypl.savefig("synapses")
    

    # Main function
    def solve(self):

        random.seed(None)
        for run in range(self.number_of_runs):

            self.record_synapses = []

            self.log_new_run(run)

            ########################################
            ######### Creating all variables #######
            ########################################
            self.input_shape = self.opt_prob.input_mapping.shape
            output_shape = self.opt_prob.output_mapping.splitted_shape
            self.n_outputs = output_shape[0]
            total_shape = self.input_shape +\
                          output_shape

            # Compute the init average error

            if("init error" in self.algo_config):
                init_average_error = self.algo_config["init error"]
            else:
                init_average_error = self.compute_init_average_error()
            print("Init average error",init_average_error)

            best_average_error = init_average_error
            self.probs = np.ones(total_shape)/2.
            average_error_neuron = np.full(
                    self.input_shape,
                    init_average_error)

            # Average error for synapses
            if(self.mode == 0):
                average_error_synapse = np.full(
                        total_shape,
                        init_average_error)
            else:
                average_error_synapse = np.zeros(
                        total_shape, dtype=float)

            counter_synapse = np.zeros(total_shape)
            counter_neuron = np.zeros(self.input_shape)
            prune_synapse = np.zeros(total_shape)
            activation_time_synapse = np.full(total_shape, 0.0)  #if this is above 1 prune away
            prune_summand = 1./(self.prune_constant*self.k_output_factor)            
            #print("prune_summand",prune_summand)
            init_n_synapse = functools.reduce(mul,output_shape[1:])
            self.n_synapses = np.full(
                    self.input_shape+(output_shape[0],),
                    init_n_synapse)
            list_indexes = [i for i in np.ndindex(
                    self.input_shape)]
            ind_len = len(list_indexes)
            half_len = ind_len//2

            


            for step in range(self.M):

                if (hasattr(self,"steps_plot")):
                    if(step in self.steps_plot):
                        if(run == 0):
                            self.record_synapses.append(np.copy(self.probs))
                        self.log_batch_error(step)

                # Check if incremental mode is active
                elif(self.incremental):
                    # Logging batch eror
                    self.log_batch_error(step,True)
                else:
                    # Logging batch eror
                    self.log_batch_error(step)
            
                if(self.incremental):
                    if(step < self.half_M):
                        # Take a random element in input space
                        i0 = list_indexes[np.random.choice(half_len)]
                    else:
                        i0 = list_indexes[np.random.choice(np.arange(half_len,ind_len))]
                
                else:
                    # Take a random element in input space
                    i0 = list_indexes[np.random.choice(len(list_indexes))]
                
                # Get the slice of synapses that are potentialy activated

                n_synapse_log = self.n_synapses[i0]
                k_input,k_output = self.get_k_input_output(i0)
                half_k_output = k_output//2

                j0,slicing_bounds,slices,iterator = self.get_output(i0,k_input,k_output)
                # Compute error
                value,exp_value,E = self.compute_error(i0,j0)

                # Update average errors and counters
                output_slicing_bounds =\
                    [tuple(
                    (max(j0[i][j]-half_k_output[i],0),
                    min(j0[i][j]+half_k_output[i]+1,output_shape[1]))\
                          for j in range(len(j0[i])) ) for i in range(len(j0))]

                it_syn = np.zeros(total_shape)
                for i in range(len(j0)):
                    input_bounds = slicing_bounds[i]
                    middle_bounds = ((i,i+1),)
                    output_bounds = output_slicing_bounds[i]
                    input_slices = self.get_slices(input_bounds)
                    middle_slices = self.get_slices(middle_bounds)
                    output_slices = self.get_slices(output_bounds)
                    slices = input_slices+middle_slices+output_slices
                    middle_output_slices = middle_slices+output_slices

                    #print("slices",slices)
                        
                    sliced_probs = self.probs[slices]
                    if(hasattr(self,'static') and self.static):
                        prune_summand= float(k_output)/self.n_synapses[i0]/self.prune_constant
            
                    prune_synapse[input_slices]= np.minimum(prune_synapse[input_slices] + prune_summand,1.1)
                    average_error_neuron[input_slices] *= (1-self.alpha)
                    average_error_neuron[input_slices] += E*self.alpha

                    activation_time_synapse[slices] = step
                    test = sliced_probs > 0.01
                    prune_synapse[slices][test] = 0.
                    
                    if(self.mode == 0):
                        average_error_synapse[slices] *= (1-self.alpha_synapse)
                        average_error_synapse[slices] += E*self.alpha_synapse
                    else:
                        average_error_synapse[slices] += float(E)/self.counter_threshold

                    counter_synapse[slices][test] += 1
                    counter_neuron[input_slices] += 1

                    test_shape_1=np.shape(test)
                    test_shape_2=test_shape_1[len(np.shape(counter_neuron)):(len(total_shape)+1)]
                    test_matrix= np.full(test_shape_2, 1.0)
                    test_matrix2=np.tensordot(counter_neuron[input_slices], test_matrix, axes=0)
                    test_neuron_counter = (test_matrix2>= self.counter_neuron_threshold)


                    #print('test shape', np.shape(test), '/n input slices shape', np.shape(input_slices), '\n test neuron shape', \
                    #    np.shape(test_neuron_counter), 'test_neuron_counter', np.shape(test_neuron_counter), 'test matrix', np.shape(test_matrix), 'test matrix2', np.shape(test_matrix2))
                    test_00 = np.logical_and(test,test_matrix2)
                    test_0 = np.logical_and(test_00,sliced_probs < 0.99)
                    E_matrix=np.full(np.shape(test_0), E)
                    if False:
                        test_1 = np.logical_and(test_0,(E_matrix.T > self.error_coeff * average_error_neuron[input_slices].T).T)
                    else:
                        test_1 = np.logical_and(test_0,(E> self.error_coeff * average_error_neuron[i0]))

                    sliced_probs[test_1] = 0.
                    
                    axes = tuple(range(len(self.input_shape)+1,len(total_shape)))
                    delta_n_synapses = np.sum(test_1,axis=axes)

                    sliced_n_syn = self.n_synapses[input_slices+middle_slices]
                    sliced_n_syn -= delta_n_synapses
                    #print("sum",np.sum(self.n_synapses))
                    counter_synapse[slices][test_0] = 0
                    if(self.mode == 1):
                        average_error_synapse[slices][test_0] = 0


                    test_minimal = np.logical_and(delta_n_synapses > 0,sliced_n_syn < self.minimal_number_synapses)
                    sliced_probs[test_minimal] = sliced_probs[test_minimal]*2
                    
#                # For each concerned synapse:
#                for i in range(len(output_slicing_bounds)):
#                    for input_neuron in iterator[i]:
#                        out_iterator = self.get_iterator(output_slicing_bounds[i])
#                        prune_synapse[input_neuron] = prune_synapse[input_neuron] + prune_summand
#
#                        average_error_neuron[input_neuron] =\
#                                average_error_neuron[input_neuron]*(1-self.alpha) +\
#                                E*self.alpha
#                        for output_neuron in out_iterator:
#
#                            activation_time_synapse[input_neuron][i][output_neuron]=step
#
#                            # If synapse still exists
#                            if(self.probs[input_neuron][i][output_neuron] > 0.01):
#
#                                prune_synapse[input_neuron][i][output_neuron]=0.0
#
#                                # Update average error
#                                if(self.mode == 0):
#                                    average_error_synapse[input_neuron][i][output_neuron] =\
#                                            average_error_synapse[input_neuron][i][output_neuron]*(1-self.alpha_synapse) +\
#                                            E*self.alpha_synapse
#                                else: 
#                                    average_error_synapse[input_neuron][i][output_neuron] += E/self.counter_threshold
#                                counter_synapse[input_neuron][i][output_neuron] += 1
#
#                                # If counter reached threshold:
#                                if(counter_synapse[input_neuron][i][output_neuron] >= self.counter_threshold and self.probs[input_neuron][i][output_neuron]<0.99):
#
#                                    # If error is bigger than average, delete synapse, and update n synapses
#                                    #counter_synapse[input_neuron,i][output_neuron] = 0
#
#                                    # Check synapse value
#                                    if(average_error_synapse[input_neuron][i][output_neuron] > self.error_coeff * average_error_neuron[input_neuron]):
#
#                                            self.probs[input_neuron][i][output_neuron] = 0.
#                                            self.n_synapses[input_neuron][i] -= 1
#                                            counter_synapse[input_neuron][i][output_neuron] = 0
#                                            if self.n_synapses[input_neuron][i]== self.minimal_number_synapses:
#                                                        self.probs[input_neuron][i]= self.probs[input_neuron][i]*2
#
#                                    # Elif average mode, reset average to 0
#                                    elif(self.mode == 1):
#                                            
#                                            average_error_synapse[input_neuron][i][output_neuron] = 0.
                if step==0:
                    checking_active_syn_time= 5* self.input_shape[0]*self.k_output_factor// min(k_output)
                    #print('checking_active_syn_time', checking_active_syn_time)
                    running_mean_error = E
                    running_mean_k = k_output
                else:

                    running_mean_k = running_mean_k*0.98 + 0.02*k_output
                    running_mean_error = running_mean_error*0.98 + 0.02*E

                if step % 50 == 49:
                    #print("min synapse",np.min(self.n_synapses))
                    
                    #print("n synapse",np.sum(self.probs > 0))
                    #print('step', step, '\n mean error', running_mean_error, '\n mean k', running_mean_k,'\n check time', checking_active_syn_time)  
                    #print('number_synapse', np.transpose(n_synapses))
                    printing=False
                    if printing:
                        for z in range(0,self.input_shape[0],75):
                            for i in range(len(output_iterators)):
                                print('Output dimension', i)
                                print('input synapse', z, '\n', self.probs[z][i], '\n counter synapse', z, '\n', counter_synapse[z][i], \
                                    '\n average error neuron',z, \
                                    np.around(average_error_neuron[z],2), '\n average error synapses', np.around(average_error_synapse[z][i],2), \
                                    '\n activation time', activation_time_synapse[z][i], '\n delete', self.input_shape[0]*self.k_output_factor/self.n_synapses[z][i])
#                prune_inactive_syn_mode_1=False       
#                if step == checking_active_syn_time and prune_inactive_syn_mode_1:
#                    prune_const=6
#                    prune_const2=10
#                    checking_active_syn_time +=  self.input_shape[0]*self.k_output_factor// min(running_mean_k)
#                    print('step', step, 'next_check_time', checking_active_syn_time)
#                    # For each concerned synapse:
#                    for i in range(len(output_slicing_bounds)):
#                        for input_neuron in iterator[i]:
#                            out_iterator = self.get_iterator(output_slicing_bounds[i])
#                            for output_neuron in out_iterator:
#
#                                # If synapse still exists
#                                if(self.probs[input_neuron][i][output_neuron] > 0.01 and self.probs[input_neuron,i][output_neuron]<0.99):
#                                    # Check synapse value
#                                        if(average_error_synapse[input_neuron,i][output_neuron] > 1.5 * average_error_neuron[input_neuron] \
#                                            and step > activation_time_synapse[input_neuron,i][output_neuron]+ prune_const*self.input_shape[0]*self.k_output_factor/n_synapses[input_neuron][i] ):
#
#                                                self.probs[input_neuron,i][output_neuron] = 0.
#                                                n_synapses[input_neuron,i] -= 1
#                                                counter_synapse[input_neuron,i][output_neuron] = 0
#                                                if n_synapses[input_neuron,i]== self.minimal_number_synapses:
#                                                        self.probs[input_neuron,i]= self.probs[input_neuron,i]*2                                                
#                                        elif(average_error_synapse[input_neuron,i][output_neuron] > 1.1* average_error_neuron[input_neuron] \
#                                            and step > activation_time_synapse[input_neuron,i][output_neuron]+ prune_const2*self.input_shape[0]*self.k_output_factor/n_synapses[input_neuron][i] ):
#
#                                                self.probs[input_neuron,i][output_neuron] = 0.
#                                                n_synapses[input_neuron,i] -= 1
#                                                counter_synapse[input_neuron,i][output_neuron] = 0                                   
#                                                if n_synapses[input_neuron,i]== self.minimal_number_synapses:
#                                                        self.probs[input_neuron,i]= self.probs[input_neuron,i]*2


                prune_inactive_syn_mode_2=True       
                if step%10 == 0 and prune_inactive_syn_mode_2:
                        #print('step', step, 'prun inactive synapses')
                        # For each concerned synapse:
#                        for input_neuron in list_indexes:
#                            for i in range(len(output_slicing_bounds)):
#                                for output_neuron in np.ndindex(output_shape[1:]):
#
#                                    # If synapse still exists
#                                    if(self.probs[input_neuron][i][output_neuron] > 0.01 and self.probs[input_neuron][i][output_neuron]<0.99):
#                                        # Check synapse value
#                                            #if(sum_time_synapse[input_neuron,i][output_neuron] > 1.0 * sum_time_neuron[input_neuron] \
#                                            #    and prune_synapse[input_neuron, i, output_neuron] > 1.0 ):
#                                            if(prune_synapse[input_neuron][i][output_neuron] >1.0 and average_error_synapse[input_neuron][i][output_neuron] > 1.0* average_error_neuron[input_neuron]):
#                                                    self.probs[input_neuron][i][output_neuron] = 0.
#                                                    self.n_synapses[input_neuron][i] -= 1
#                                                    counter_synapse[input_neuron][i][output_neuron] = 0
#                                                    if self.n_synapses[input_neuron][i]== self.minimal_number_synapses:
#                                                            self.probs[input_neuron][i]= self.probs[input_neuron][i]*2                                                    
                    
                        # Rewriting with matrices 
                        test1 = self.probs > 0.01
                        test2 = self.probs < 0.99
                        test3 = prune_synapse > 1.0

                        test4 = (average_error_synapse.T > average_error_neuron.T).T
                        #test = np.logical_and(test1,np.logical_and(test2,np.logical_and(test3,test4)))
                        test = np.logical_and(test1,np.logical_and(test2,test3))
        
                        self.probs[test] = 0.
                        counter_synapse[test] = 0

                        # Count number of 
                        axes = tuple(range(len(self.input_shape)+1,len(total_shape)))
                        delta_n_synapses = np.sum(test,axis=axes)
                        
                        #print('Number remaining',np.sum(self.probs > 0))
                        self.n_synapses -= delta_n_synapses

                        test_minimal = np.logical_and(delta_n_synapses > 0,self.n_synapses < self.minimal_number_synapses)
                        self.probs[test_minimal] = self.probs[test_minimal]*2

                        
                        if step+1 % 50000 == 0:
                            print('num synapses', self.n_synapses)

                    #aa=np.ones(total_shape)*step - np.tensordot(prune_const* self.input_shape[0]*self.conf_coeff/n_synapses, np.ones(output_shape[1]),axes=0)
                    #probs = np.multiply(probs, (activation_time_synapse > aa))
                    #print('check',checking_active_syn_time)
                    #print('aa',aa)
                    #print('activation_time', activation_time_synapse )

                if step%100==49:  
                    a=step//50  +10
                    if False: #plot probs for one dim functions
                        file_name= self.name+'_probs_plot_'+str(a)
                        self.save_print_probs(self.probs,self.input_shape,file_name)
                        file_name= self.name+'_prune_plot_'+str(a)
                        self.save_print_probs(prune_synapse,self.input_shape,file_name)

                    if False:

                        x1=15
                        x2=15
                        probs_plot= self.probs[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_probs_plot_'
                        pypl.imshow(probs_plot)
                        pypl.savefig(file_name)
                        prune_plot= prune_synapse[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_prune_plot_'
                        pypl.imshow(prune_plot)
                        pypl.savefig(file_name)
                        x1=20
                        x2=20
                        probs_plot= self.probs[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_probs_plot_'
                        pypl.imshow(probs_plot)
                        pypl.savefig(file_name)
                        prune_plot= prune_synapse[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_prune_plot_'
                        pypl.imshow(prune_plot)
                        pypl.savefig(file_name)
                        x1=25
                        x2=25
                        probs_plot= self.probs[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_probs_plot_'
                        pypl.imshow(probs_plot)
                        pypl.savefig(file_name)
                        prune_plot= prune_synapse[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_prune_plot_'
                        pypl.imshow(prune_plot)
                        pypl.savefig(file_name)
                        x1=10
                        x2=10
                        probs_plot= self.probs[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_probs_plot_'
                        pypl.imshow(probs_plot)
                        pypl.savefig(file_name)
                        prune_plot= prune_synapse[x1, x2, 0, :, :]
                        file_name= self.name+ str(x1)+'_'+str(x2)+'_'+str(a)+'_prune_plot_'
                        pypl.imshow(prune_plot)
                        pypl.savefig(file_name)                        
                        #self.save_print_probs(probs_plot,self.input_shape,file_name)
                        #prune_plot= self.probs[25, :, 0, :, 25]
                        #file_name= self.name+'_prune_plot_'+str(a)
                        #self.save_print_probs(prune_plote,self.input_shape,file_name)                        
                #if step%20==0:
                #    self.print_probs(probs,self.input_shape)



            if(run == 0 and self.plot_synapses):
                self.plot_synapse_history()

            self.log_test_cross_error()

            # Log final results
            self.log("\n",logging.DEBUG)
            self.log("Counter synapses:",logging.DEBUG)
            self.log(counter_synapse[-2,0],logging.DEBUG)
            self.log("\n",logging.DEBUG)
        return 