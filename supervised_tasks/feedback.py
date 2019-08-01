# Implementation of feedback mechanisms
# Author: Raphaël Dang-Nhu  
# Date: 23/04/2019

import math
import algorithms
import numpy as np

class Feedback:

    def __init__(self,feedback_config):
        self.config = feedback_config

class L1Feedback(Feedback):
    
    def get_error(self,value,expected_value):

        assert(len(value) == len(expected_value))
        
        res = 0.
        for (v,e) in zip(value,expected_value):
            res += abs(v-e)
        return res

class L2Feedback(Feedback):
    
    def get_error(self,value,expected_value):

        assert(len(value) == len(expected_value))
        
        res = 0.
        for (v,e) in zip(value,expected_value):
            res += (v-e)*(v-e)

        return res

class LNFeedback(Feedback):
    
    def __init__(self,config):
        Feedback.init(self,config)
        self.N = float(self.config["N"])

    def get_error(self,value,expected_value):

        assert(len(value) == len(expected_value))
        
        res = 0.
        for (v,e) in zip(value,expected_value):
            res += math.pow(abs(v-e),self.N)

        return res




