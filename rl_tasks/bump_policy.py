# Policy with bumps
# Author: Raphael Dang-Nhu
# 18/05/2019

import json
import matplotlib.pyplot as pypl
import collections
import logging
import datetime
import numpy as np
import gym
from gym import spaces
import tensorflow as tf
import sys, os
from gym_project_conf import *
import policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_functions'))
import algorithms

class BumpPolicy(policy.Policy):
    
    def subpolicy_init(self):
        self.k_in = self.config['k in']
        self.k_out = self.config['k out']
        self.half_k_in = self.k_in //2
        self.half_k_out = self.k_out //2
        print("k_in",self.k_in)
        print("half k in",self.half_k_in)
        print("k_out",self.k_out)
        print("half k out",self.half_k_out)
        print("\n")
        self.k_array_in = np.array([self.k_in])
        self.k_array_out = np.array([self.k_out])
        return

    #######################################################
    #### Update functions for case wthout value funcs #####
    #######################################################

    def update_synapses(self):
            
           #print("Reward for this episode",self.total_reward)
           #print("Mean reward",self.mean_reward)
           #print("\n")

           #self.print_probs(self.probs,self.input_shape)

           self.iterate_fixed_k(synapse_update_function,self.half_k) 

           
           return

    def update_reward(self) :
       self.mean_reward = (1 - self.alpha) * self.mean_reward + self.alpha * self.total_reward

    #######################################################
    #### Update functions for the value function case #####
    #######################################################

    def synapse_update_matrix(self,delta,i0,j0):

       coeff = self.coeff
       slices =\
            tuple(\
                  [slice(max(i0[i]-self.half_k_in,0),
                   min(i0[i]+self.half_k_in+1,self.input_shape[0]))\
                  for i in range(len(i0))]+\
                  [slice(0,1)]+\
                  [slice(max(j0[i]-self.half_k_out,0),
                    min(j0[i]+self.half_k_out+1,self.output_shape[1]))\
                    for i in range(len(j0))])

       #print("j0",j0)
       #print("slices",slices)
        
       if(delta > 0.):
           scalar = min(coeff*delta,1) 
           #print("probs",self.probs[slices])
           self.probs[slices] = self.probs[slices] + (0.9 - self.probs[slices])*scalar

       else:
           scalar = max(coeff*delta,-1) 
           self.probs[slices] = self.probs[slices] + (self.probs[slices]-0.1)*scalar

    def synapse_update_function_delta(self,proba_synapse):
        
       delta = self.delta
       if(delta > 0.):
           result = proba_synapse + (0.9 - proba_synapse) * min(coeff * delta,1)
       else:
           result = proba_synapse + (proba_synapse - 0.1) * max(coeff * delta,-1)
       return result

    def store_reward(self,obs,rew,j0):

        self.rewards.append(rew)
        #print("obs",obs)
        #print("neuron",self.get_neuron(obs))

        self.obs.append(self.get_neuron(obs))
        self.actions.append(j0)
        
        
    def policy_update(self,offset=-1):

        # Check if there are at least n+1 
        if((offset == -1 and len(self.rewards) >= self.config["n steps"]+1) or len(self.rewards) +offset >= self.config["n steps"]+1):
            
            update_index,delta = self.update_value_function(offset)
            #print("delta",delta)

            # Write in list_activations to use inner k function
            action_update_index = update_index + 1
            if(action_update_index < len(self.rewards)):
                i0 = self.obs[update_index]
                j0 = self.actions[action_update_index][0]
                self.synapse_update_matrix(delta,i0,j0)
                #print("before",self.probs[tuple(i0)][0][j0])
                #self.list_activations= [(i0,j0)]
                #self.iterate_fixed_k(self.synapse_update_function_delta,self.half_k) 
                #self.list_activations = []
#                print("Action update innex",action_update_index)
#                print("after",self.probs[tuple(i0)][0][j0])


    #######################################################
    ##### Get next action with the current policy #########
    #######################################################
    def get_action(self,observation):
        
        # Get input neurons that are closest to observation
        #print("obs",type(observation[0]))
        #print("ints",self.intervals)
        #print("left",self.left_bounds)

        activated_neuron = self.get_neuron(observation)
        #print("activated neuron",activated_neuron)

        # GET THE OUTPUT
        j0,_,_,_ = self.get_output(activated_neuron,self.k_array_in,self.k_array_out)

        # Return corresponding output
        #print("j0",j0)
        output = np.array(j0) * self.output_interval + self.olb
    
        if(not self.config["value function"]):
            self.list_activations.append((activated_neuron,j0))


        # Do some logging
        #self.log("GET ACTION",logging.DEBUG)
        #self.log("Observation: " + str(activated_neuron),logging.DEBUG)
        #self.log("Output value: " + str(output),logging.DEBUG)
        #self.log("Probabilities: ",logging.DEBUG)
        #self.log(self.probs,logging.DEBUG)
        return j0,output
