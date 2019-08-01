# Opimization algorithms
# Author: Raphael Dang-Nhu
# 10/03/2019

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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_functions'))
import algorithms

# Class to represent a learning policy for a RL environment (inherits algorithm to get the tensorflow graph)
class Policy(algorithms.Algorithm):
    
    # Construct policy for an environment
    def __init__(self,env_name,config,common_config,debug_file,perf_file):

        self.env = gym.make(env_name)
        self.env.reset()
        self.config = config
        self.common_config = common_config
        self.nruns = common_config["runs"]
        self.neps = common_config["eps"]

        self.coeff = config['alpha policy']
        self.nin = config['nin_per_dim']
        self.non = config['non']
        self.alpha = config["alpha reward"]

        # Generate tensor of synapse probabilities
        act_space = self.env.observation_space

        if("left bounds" in config):
            self.left_bounds = np.array(config["left bounds"])
        else:
            self.left_bounds = np.array([max(aux,-3.) for aux in self.env.observation_space.low])
        if("right bounds" in config):
            self.right_bounds = np.array(config["right bounds"])
        else:
            self.right_bounds = np.array([min(aux,3.) for aux in self.env.observation_space.high])
        self.intervals = (self.right_bounds - self.left_bounds)/float(self.nin-1)

        if(isinstance(act_space,spaces.Box)):
            # Complete the branching here
            self.input_dim = len(self.intervals) 
            self.output_dim = 1
        else:
            assert(False)

        self.input_shape = (self.nin,)*self.input_dim
        self.output_shape = (self.output_dim,self.non,)
        self.total_shape = self.input_shape + self.output_shape 

        self.olb = self.env.action_space.low
        self.orb = self.env.action_space.high
        self.output_interval = (self.orb - self.olb)/float(self.non-1)

        print("Intervals",self.left_bounds,self.right_bounds)
        print("Output Interval",self.olb,self.orb)

        self.debug_logger = algorithms.setup_logger("debug",debug_file,logging.DEBUG)
        self.perf_logger = algorithms.setup_logger("perf",perf_file,logging.INFO)

        # Log the problem configuration
        self.log("PROBLEM CONFIGURATION",logging.INFO)
        self.log("Env name: "+env_name,logging.INFO)
        self.log("ALGORITHM CONFIGURATION",logging.INFO)
        self.log(json.dumps(self.config,indent=4),logging.INFO)

    def run_init(self):
        self.probs = np.ones(self.total_shape)/2

       
        if(self.config["value function"]):
            # Create value function tabular
            self.state_values = np.full(self.input_shape,self.config['init value'],dtype=float)
            self.n_steps = self.config["n steps"]

        # Create the variables to update the probas
        self.mean_reward = 0.
        
        # Init figs
        if("save" in self.common_config):
            self.id = str(datetime.datetime.now())
            folder = gym_fig_folder + self.id + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Dump config
            with open(folder+"config.txt",'w') as f:
                f.write(json.dumps(self.config,indent=4))

        # Call the initialization procedure of each particular policy
        self.subpolicy_init()

    # Function called at the beginning of each episode
    def start_episode(self):

        if(self.config["value function"]):
            self.rewards = []
            self.obs = []
            self.actions = []
            self.obs_continuous = []
        else:
            self.list_activations = []

        self.total_reward = 0.
    

    # Function to get a neuron from observation
    def get_neuron(self,observation):

        observation_aux = []
        for i in range(len(observation)):
            if(isinstance(observation[i],np.ndarray)):
                observation_aux.append(observation[i][0])
            else:
                observation_aux.append(observation[i])
                
        res = ((observation_aux +\
                self.intervals/2. - self.left_bounds)//self.intervals ).astype(int)

        for i in range(len(res)):
            res[i] = np.clip(res[i],0,self.nin-1)

        return res

    # Function to update tabular value function
    def update_value_function(self,offset):

        # Sum the rewards in the array
        update_index = len(self.rewards) - self.config["n steps"] - 1
        if(offset != -1):
            update_index += offset
        left_index = update_index + 1
        sum_reward = sum(self.rewards[left_index:])

        # Bootstrap with current state
        if(offset <= 0):
            new_value = sum_reward + self.state_values[tuple(self.obs[-1])]
        else:
            new_value = sum_reward
#        print("Update idnex",update_index)
#        print("new_value",new_value)

        # Update last state
        last_state = tuple(self.obs[update_index])
        delta = new_value - self.state_values[last_state]
        self.delta = delta
        #print("Old state value",self.state_values[last_state])


        # Also update neighbouring states
#        
#        slicing_bounds =\
#            tuple(\
#                  (max(last_state[i]-self.half_k,0),
#                   min(last_state[i]+self.half_k+1,self.input_shape[0]))\
#                  for i in range(len(last_state)))
#
#        iterator = self.get_iterator(slicing_bounds)
#
#                
#        # For each input neuron
#        for input_neuron in iterator:
#
#            self.state_values[input_neuron] = self.state_values[input_neuron] +\
#                self.config["alpha value function"] * delta

        # Do not update neighbours
        self.state_values[last_state] = self.state_values[last_state] +\
                self.config["alpha value function"] * delta

#        if(sum_reward > 50 ):
#            print("\n")
#            print("Offset",offset)
#            print("Reward",new_value)
#            print("Last state",last_state)
#            print("delta",delta)
#            print("New state value",self.state_values[last_state])
#            print("\n")
       
        return update_index,delta


    ###########################################################
    ############## Printing functions #########################
    ###########################################################

    def print_state_values(self,save=False,episode=0):
        pypl.figure()
        pypl.imshow(self.state_values)
        pypl.colorbar()
        if(save):
            folder = gym_fig_folder+self.id+"/values/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            pypl.savefig(folder+"episode_"+str(episode))
        else:
            pypl.show()
        pypl.close()
                 
    def train(self):
        neps=self.neps
        nsteps = 10000

        for run in range(self.nruns):
    
            self.run_init()
            self.log_new_run(run)

            # Iterate over episodes
            for episode in range(neps):

                #print("Episode",episode)

                self.start_episode()
                obs = self.env.reset()
                if(self.config["value function"]):
                    # This action is not relevant, it is as the 0 reward
                    j0,_ = self.get_action(obs)
                    self.store_reward(obs,0,j0)

                for step in range(1,nsteps+1):
                   
                    # Comment this line to avoid showing the video
                    #self.env.render()

                    # Sample new action every 
                    j0,action = self.get_action(obs)

                    obs, rew, done, info = self.env.step(action)

                    # Store reward
                    if(self.config["value function"]):
                        self.store_reward(obs,rew,j0)

                    self.total_reward += rew

                    if done:

                        # Update if value function
                        if(self.config["value function"]):
                            for i in range(0,self.config["n steps"]+1):
                                self.policy_update(offset=i)
                        else: 
                            if(episode > 10):
                                self.update_synapses()
                            self.update_reward()

                        #print("Total Reward",self.total_reward)
                        self.log("Episode: "+str(episode),logging.INFO)
                        self.log("Reward: "+str(self.total_reward),logging.INFO)
                        break
                    else:
                        # Update if value function
                        if(self.config["value function"]):
                            self.policy_update()
                
                # Uncomment to plot value function for the moutain car
                #self.print_state_values(save=True,episode=episode)
                # Uncomment to plot probabilities. If the input dimension is more than two, a few points are selected for the first dimensions.
                #self.print_probs(self.probs,self.input_shape,save=True,episode=episode)

                #if(episode == neps-1):
            #self.print_probs(self.probs,self.input_shape)



