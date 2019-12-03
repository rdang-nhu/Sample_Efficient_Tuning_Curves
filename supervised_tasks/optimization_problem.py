# Class representing an optimization problem in the neuronal form
# Author: Raphael Dang-Nhu
# 28/02/2019

import numpy as np
import json
from utils import *

class OptimizationProblem:

    # Constructor reads a json file describing the experimental configuration
    def __init__(self,function_config,mapping_config,feedback_config):
        
        self.mapping_config = mapping_config
        self.function_config = function_config
        
        # Extract functions and mappings
        self.function, self.input_mapping, self.output_mapping = \
            config_extract_function(self.function_config,self.mapping_config)
        
        # Extract feedbacks
        self.feedback = config_extract_feedback(feedback_config)
            
        input_shape = self.input_mapping.shape
        output_shape = self.output_mapping.shape
        output_dim = self.function_config["output dim"]

        self.expected_output_neuron = np.zeros(shape=input_shape)
        self.input_neuron_value = self.input_mapping.get_neuron_value()
        self.output_neuron_value = [self.output_mapping.get_neuron_value(),]
        
        # Compute expected output value
        self.expected_output_value = np.zeros(shape=(input_shape+(output_dim,)))
        for index in np.ndindex(input_shape):
            input_val = [self.input_neuron_value[i][index[i]] for i in range(len(index))]
            expected_output_val = self.function(input_val)
            self.expected_output_value[index] = np.array(expected_output_val)

            # Check that evertything is inside the bounds
            #assert(expected_output_val >= outlb)
            #assert(expected_output_val <= outrb)

        return

