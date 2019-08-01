# Functions and mappings
# Author: Raphaël Dang-Nhu  
# Date: 08/04/2019

import functools
from operator import mul
import math
import algorithms
import numpy as np


# List of common functions that can be used in optimization problems
class Function:

    def __init__(self,config):
        self.config = config
        config["input dim"] = self.input_dim
        config["output dim"] = self.output_dim

# Class for one dimensional functions
class OneDimFunction(Function):
    
    def __init__(self,config):
        self.input_dim = 1
        self.output_dim = 1
        Function.__init__(self,config)
    

class Identity(OneDimFunction):

    # Make the classs callable
    def __call__(self,x):
        return x[0]

class Linear(OneDimFunction):

    # Make the classs callable
    def __call__(self,x):
        return self.config['linear_coeff']*x

class Affine(OneDimFunction):

    # Make the classs callable
    def __call__(self,x):
        return self.config['a']*x+self.config['b']

class Polynomial(OneDimFunction):

    def __init__(self,config):
        OneDimFunction.__init__(self,config)
        # coeffs is a list of values [a_(n-1),...,a_0] such that P(X) = a_n X^n + ... a_0 X^0 
        self.coeffs = self.config['coeffs']

    def __call__(self,x):
        n = len(self.coeffs)
        res = 0
        for i in range(0,n):
            res = res * x[0] + self.coeffs[i]
        return res
        
class Exp(OneDimFunction):

    def __call__(self,x):
        return math.exp(x)

class Log10(OneDimFunction):

    def __call__(self,x):
        return math.log10(x)

class Sqrt(OneDimFunction):

    def __call__(self,x):
        return math.sqrt(x)

class Sinh(OneDimFunction):

    def __call__(self,x):
        return math.sinh(x)

class Sin(OneDimFunction):

    def __call__(self,x):
        return math.sin(x[0])

class Sin2(OneDimFunction):

    def __call__(self,x):
        return math.sin(3.14*x[0]) /2  +0.25          

class Tan(OneDimFunction):

    def __call__(self,x):
        return math.tan(x)

class Arcsin(OneDimFunction):

    def __call__(self,x):
        return math.asin(x)

# Rectified linear function
# Coeffs are the two linear coeffs
# Threshold is the value where the linear coeff changes
# The function is continuous
class Relu(OneDimFunction):

    def __init__(self,config):
        Function.__init__(self,config)
        self.coeffs = self.config['coeffs']
        self.threshold = self.config['threshold']
        self.constant = self.coeffs[0]*self.threshold

    def __call__(self,x):
        if(x <= self.threshold):
            return self.coeffs[0]*x 
        else:
            return self.constant + self.coeffs[1] * (x - self.threshold)

#####################################
# Classes for one dimensional output#
#####################################
class OneDimOutputFunction(Function):
    
    def __init__(self,config):
        self.input_dim = config['input dimension'] 
        self.output_dim = 1
        Function.__init__(self,config)

class Sum(OneDimOutputFunction):

    # Make the classs callable
    def __call__(self,x):
        return sum(x)

class Product(OneDimOutputFunction):

    # Make the classs callable
    def __call__(self,x):
        return functools.reduce(mul,x,1)

class First(OneDimOutputFunction):

    # Make the classs callable
    def __call__(self,x):
        return x[0]

class Second(OneDimOutputFunction):

    # Make the classs callable
    def __call__(self,x):
        return x[1]

# Map function name to class reference
name_to_ref = {'identity': Identity, 'linear':Linear,'affine':Affine,
        'polynomial':Polynomial,'sin':Sin,'tan':Tan,'arcsin':Arcsin,'sinh':Sinh,"sin2":Sin2,
        'exp':Exp,'log10':Log10,'sqrt':Sqrt,'relu':Relu,'sum':Sum,'product':Product,'first':First,'second':Second}

# Multidimensional function
class MultiDimFunction(Function):

    def __init__(self,config):
        if('input dimension' in config):
            self.input_dim = config['input dimension']
        else:
            print("No input dimension specified, taking 1")
            self.input_dim = 1
        self.output_dim = len(config['components'])
        Function.__init__(self,config)

        self.components = []
        # Read all functions
        for conf in self.config['components']: 
            conf['input dimension']=self.input_dim
            class_ref = name_to_ref[conf['name']]
            comp = class_ref(conf)
            self.components.append(comp)
        

    def __call__(self,x):
        
        a =[ comp(x) for comp in self.components]
        return a

# Add multiD to name_to_ref
name_to_ref["multi dimensional"] = MultiDimFunction

# List of mappings that can be used in optimization problems
class Mapping:

    def __init__(self,function_config,config,mode,index=0):
        # function config, in case some of the parameters must be derived from the function
        self.function_config = function_config
        # mapping config
        self.config = config
        self.index = index
        config['dimension'] = self.dim

    def get_bounds(self):
        if(self.mode == "input"):
            
            if(isinstance(self.function_config['input bounds'][0],list)):
                return self.function_config['input bounds'][self.index]
            return self.function_config['input bounds']
        else:
            return self.function_config['output bounds']
    

class Linear_Mapping(Mapping):

    def __init__(self,function_config,config,mode,index=0):
        self.dim = function_config["output dim"]
        Mapping.__init__(self,function_config,config,mode)
        self.shape = (self.config['n'],)
        self.splitted_shape = (self.dim,self.config['n'])
        self.mode = mode

    def get_neuron_value(self):

        nin = self.config['n']
        neuron_value = np.zeros(self.shape)
        

        [lb,rb] = self.get_bounds()
        interval = (rb-lb)/float(nin-1)
        value = lb
        for i in range(nin):

            #expected_output_neuron[i]  = (expected_output_val - outlb + output_interval/2.)//output_interval
            neuron_value[i] = value

            value += interval
        return neuron_value


class Relu_Mapping(Mapping):

    def __init__(self,function_config,config,mode,index=0):
        self.dim = function_config["output dim"]
        Mapping.__init__(self,function_config,config,mode)
        self.shape = (self.config['n'],)

    def get_neuron_value(self):

        nn = self.config['n']
        
        # switch on parameters
        parameters = self.config['parameters']
        if(parameters == "same as function"):
            coeffs = self.function_config['coeffs']
            neuron_threshold = 500
            #threshold = self.function_config['threshold']

        else:
            coeffs = self.config['coeffs']
            neuron_threshold = self.config['neuron_threshold']

        neuron_value = np.zeros(self.shape)
        # Unit interval 
        [lb,rb] = self.get_bounds()
        mean_coeff = (coeffs[0] + coeffs[1])/2.
        unit_interval = (rb-lb)/(mean_coeff*float(nn-1))
        value = lb

        # Handle the first half
        left_interval = coeffs[0] * unit_interval
        for i in range(neuron_threshold):
            
            #expected_output_neuron[i]  = (expected_output_val - outlb + output_interval/2.)//output_interval
            neuron_value[i] = value
            value += left_interval

        # Handle the second half
        right_interval = coeffs[1] * unit_interval
        for i in range(neuron_threshold,nn):
            neuron_value[i] = value
            value += right_interval

        return neuron_value

# Two dimensional mapping. Neuron [angle,speed] has value true_speed^2 * sin(2*true_angle)
class Throw_Ball_Mapping(Mapping):
    
    def __init__(self,function_config,config,mode,index=0):
        self.dim = 2
        Mapping.__init__(self,function_config,config,mode)
        self.shape = (self.config['n'],)*2
        if("grid" in config and config["grid"] == "true"):
            self.splitted_shape = (1,)+self.shape
        else:
            self.splitted_shape = (2,self.config['n'])
        return
    
    def get_neuron_value(self):
        nn = self.config['n']
        
        neuron_value = np.zeros(self.shape)
    
        [[angle_left_bound,angle_right_bound],[speed_left_bound,speed_right_bound]] = self.config['bounds']
        angle_interval = (angle_right_bound-angle_left_bound)/float(nn-1)
        speed_interval = (speed_right_bound-speed_left_bound)/float(nn-1)
        angle_value = angle_left_bound
        for i in range(nn):
            speed_value = speed_left_bound
            for j in range(nn):

                #expected_output_neuron[i]  = (expected_output_val - outlb + output_interval/2.)//output_interval
                neuron_value[i][j] = speed_value ** 2 * math.sin(2*angle_value)
                speed_value += speed_interval

            angle_value += angle_interval
        
        return neuron_value

# Two dimensional mapping. Neuron [angle,speed] has value true_speed^2 * sin(2*true_angle)
class Reach_Mapping(Mapping):
    
    def __init__(self,function_config,config,mode,index=0):
        self.dim = 2
        Mapping.__init__(self,function_config,config,mode)
        self.shape = (self.config['n'],)*2
        if("grid" in config and config["grid"] == "true"):
            self.splitted_shape = (1,)+self.shape
        else:
            self.splitted_shape = (2,self.config['n'])
        return
    
    def get_neuron_value(self):
        nn = self.config['n']
        d1 = self.config['length long arm']
        d2 = self.config['length short arm']
        
        neuron_value = np.zeros(self.shape+(2,))
    
        [[alpha_left_bound,alpha_right_bound],[beta_left_bound,beta_right_bound]] = self.config['bounds']
        alpha_interval = (alpha_right_bound-alpha_left_bound)/float(nn-1)
        beta_interval = (beta_right_bound-beta_left_bound)/float(nn-1)
        alpha_value = alpha_left_bound
        for i in range(nn):
            beta_value = beta_left_bound
            for j in range(nn):

                #expected_output_neuron[i]  = (expected_output_val - outlb + output_interval/2.)//output_interval
                neuron_value[i][j] = [ d1 * math.cos(alpha_value) + d2 * math.cos(alpha_value+beta_value),
                        d1 * math.sin(alpha_value) + d2 * math.sin(alpha_value+beta_value) ]
                beta_value += beta_interval

            alpha_value += alpha_interval
        return neuron_value

# Map mapping name to class reference
mapping_to_ref = {'linear' : Linear_Mapping,'relu' : Relu_Mapping,'throw ball' : Throw_Ball_Mapping,'reach':Reach_Mapping }

# Wrapper for multidimensional mapping
class Multi_Dim_Mapping(Mapping):

    def __init__(self,function_config,config,mode):
        
        self.mode = mode

        if(self.mode == "input"):
            self.dim = function_config["input dim"]
        elif(self.mode == "output"):
            self.dim = function_config["output dim"]
        else:
            raise ValueError("Unvalid mode for mapping")
        
        self.components = []

        # Init all dimensions
        comp_type = config["type"]
        for i in range(self.dim):
            if(self.mode == 'output'):
                comp_function_config = function_config["components"][i]
                comp_function_config['input bounds'] = function_config["input bounds"]
                self.components.append(mapping_to_ref[comp_type](comp_function_config,config,mode,i))
            else:
                self.components.append(mapping_to_ref[comp_type](function_config,config,mode,i))
            
        Mapping.__init__(self,function_config,config,mode)
        self.shape = (self.config['n'],)*self.dim
        self.splitted_shape = (self.dim,self.config['n'])
        return

    def get_neuron_value(self):

        res = np.array([comp.get_neuron_value() for comp in self.components ])

        return res

# Adding multi dim to mapping to ref
mapping_to_ref["multi dimensional"] = Multi_Dim_Mapping
