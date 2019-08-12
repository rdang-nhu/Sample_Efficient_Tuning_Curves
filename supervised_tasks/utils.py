# Utility functions
# Author: Raphael Dang-Nhu
# 28/02/2019

import itertools as it
import math
import algorithms
from functions import *
from feedback import *
import multi_flo_keep_synapses
import multi_layer_perceptron
import final_algorithm.FinalAlgorithm
import reinforce


# Map feedback name to class reference
feedback_to_ref = {'L1' : L1Feedback,'L2':L2Feedback,'LN':LNFeedback}

# Extract output mappings from config (see Optimization_Problem class)
def config_extract_function(function_config,mapping_config):
    
    # Get function
    class_ref = name_to_ref[function_config['name']]
    function = class_ref(function_config)

    # Get mappings
    inp_config = mapping_config['input']
    out_config = mapping_config['output']

    # All input mappings are now multidimensional
    input_mapping = mapping_to_ref["multi dimensional"](function_config,inp_config,"input")
    
    # Split out mapping if throw ball
    if(function_config['name'] == "multi dimensional" and out_config["type"] != "reach" ):
        output_ref = mapping_to_ref["multi dimensional"]
    else:
        output_ref = mapping_to_ref[out_config['type']]
    output_mapping = output_ref(function_config,out_config,"output")
    
    return function, input_mapping, output_mapping
    #return input_neuron_value,  output_neuron_value , expected_output_neuron, expected_output_value, 


# Extract feedback from config (see Optimization_Problem class)
def config_extract_feedback(feedback_config):
    
    # Get function
    class_ref = feedback_to_ref[feedback_config['name']]
    feedback = class_ref(feedback_config)
    
    return feedback


# Function to get algorithm class
def name_to_class_ref(name,opt_problem):
    if(name.startswith("MultiFloKeepSynapses")):
        return multi_flo_keep_synapses.MultiFloKeepSynapses                     
    elif(name.startswith("FinalAlgorithm")):
        return final_algorithm.FinalAlgorithm
    elif(name.startswith("perceptron")):
        return multi_layer_perceptron.Perceptron
    elif(name == "reinforce"):
        return reinforce.Reinforce
    else: 
        raise ValueError("Algorithm does not exist")

# Extract list of algorithms
def get_algorithm(x,common_config,mappings,opt_problem,debug_file,perf_file):
    res = name_to_class_ref(x['name'],opt_problem)(x,common_config,mappings,opt_problem,debug_file,perf_file)
    return res


