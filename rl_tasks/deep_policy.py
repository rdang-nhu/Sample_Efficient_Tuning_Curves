
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
import keras
from keras.layers import Dense

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_functions'))
import algorithms

class DeepPolicy(policy.Policy):
    
    def subpolicy_init(self):

        # Create an actor network
        input_layer = keras.Input(shape=(self.input_dim,))
        h0_layer = Dense(self.config["width layers"],activation='relu')(input_layer)
        h1_layer = Dense(self.config["width layers"],activation='relu')(h0_layer)
        self.output = Dense(self.non,activation='softmax')(h1_layer)
        self.model = keras.Model(input=input_layer,output=self.output)
    
        # Get the weights (we use a custom optimization procedure here)
        self.model_weights = self.model.trainable_weights

        self.model.summary()
        print(self.model_weights)

        # Get the gradients
        output_one_hot = keras.Input(shape=(1,),dtype=tf.int32)
        grads = keras.backend.gradients(\
                # Custom loss
                -self.output[0][output_one_hot[0][0]],
                # Variables
                self.model_weights)

        self.function1 = keras.backend.function([input_layer,output_one_hot],grads)
        self.function2 = keras.backend.function([input_layer,output_one_hot],[self.output[0][output_one_hot[0][0]]])

        self.clip = self.config["clip"]
        return

    def store_reward(self,obs,rew,j0):

        self.rewards.append(rew)
        #print("obs",obs)
        #print("neuron",self.get_neuron(obs))
        
        self.obs_continuous.append(obs)
        self.obs.append(self.get_neuron(obs))
        self.actions.append(j0)

        
    def policy_update(self,offset=-1):
        
        # Check if there are at least n+1 
        if((offset == -1 and len(self.rewards) >= self.config["n steps"]+1) or len(self.rewards) +offset >= self.config["n steps"]+1):
            
            update_index,delta = self.update_value_function(offset)

            # Write in list_activations to use inner k function
            action_update_index = update_index + 1
            if(action_update_index < len(self.rewards)):
        

                obs = self.obs_continuous[update_index]
                #print("Updated obs",obs)
                j0 = self.actions[action_update_index]

                # Update weights with reinforce rule
                grads = self.function1([[obs],[[j0]]])
                probas = self.function2([[obs],[[j0]]])


                # Multiply by delta/proba
                scaled_grads_old = self.coeff*delta*np.array(grads)/max(probas[0],0.0001)
    
                scaled_grads = [np.clip(aux,-self.clip,self.clip) for aux in scaled_grads_old]
                old_max = np.amax(np.abs(scaled_grads_old[0]))
                new_max = np.amax(np.abs(scaled_grads[0]))

                if(new_max < old_max):
                    print("Max grad bef",old_max)
                    print("Max grad af",new_max)
                    print("\n")

                # Update weights                
                weights = np.array(self.model.get_weights())
                self.model.set_weights(weights+scaled_grads)

                #print(grads[0][0])
                #print(probas)
#                print("\n")
#                print("Old weights",weights[0][0])
#                print("\n")
#                print("Grads",scaled_grads[0][0])
#                print("\n")
#                print("New weights",self.model.get_weights()[0][0])
#                print("\n")
                

    #######################################################
    ##### Get next action with the current policy #########
    #######################################################
    def get_action(self,observation):
        
        # Feed obs to neural network
        #print("Observation",observation.shape)
        output = self.model.predict(np.array([observation]))

        # GET THE OUTPUT
        #print("Output",output)
        jmax = np.argmax(output) 
        j0 = np.random.choice(len(output[0]),1,p=output[0])[0]
        #print("\n")
        #print("Observation",observation)
        #print("j0",j0)
        #print("Max output",output[0][jmax])
        
        # See second output
        output = np.array(j0) * self.output_interval + self.olb

        # Do some logging
        self.log("GET ACTION",logging.DEBUG)
        self.log("Observation: " + str(observation),logging.DEBUG)
        self.log("Output value: " + str(output),logging.DEBUG)
        self.log("Probabilities: ",logging.DEBUG)
        self.log(self.probs,logging.DEBUG)
        return j0,output
