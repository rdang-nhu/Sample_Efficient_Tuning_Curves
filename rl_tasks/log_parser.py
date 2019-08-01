# Functions to parse the logs
# Author: Raphael Dang-Nhu
# 09/04/2018

from ast import literal_eval as make_tuple
from gym_project_conf import *
import json
import numpy as np
from collections import defaultdict

class LogParser:

    def __init__(self):
        self.window = 500
        return
    
    def convolve(self,array):
        return np.convolve(array,np.ones((self.window,))/self.window,'valid')


    # Class to read a line
    class line_reader:
        
        def __init__(self,line):
            self.line = line

        def float(self,index):
            a = self.line[index].rstrip()
            if(len(a) > 0 and a[0] == "["):
                a = a[1:]
            if(len(a) >  0 and a[-1] == "]"):
                    a = a[:-1]
            if(len(a) > 0):
                return float(a)
            return -1

        def int(self,index):
            a = self.line[index].rstrip()
            if(len(a) > 0 and a[0] == "["):
                a = a[1:]
            if(len(a) >  0 and a[-1] == "]"):
                    a = a[:-1]
            if(len(a) > 0):
                return int(a)
            return -1

            

        def tuple(self,index):
            return make_tuple(self.line[index].rstrip())

    # Function to read a log file. Returns a dictionary with all read info
    def parse_log(self,file_name,config):

        # Create dict to populate with data
        self.n_runs = config["common"]["runs"]
        self.n_eps = config["common"]["eps"]
        data = [ {"reward":[]} for i in range(self.n_runs)]

        # Open files
        with open(file_name,"r") as f:
            
            lines = f.readlines()
            
            # Gathered info
            run = -1

            i = 0
            while(i < len(lines)):
                sl = lines[i].split(" ")
                r = self.line_reader(sl)
                if(len(sl) > 0):
                    # reading steps
                    # RUN
                    if(sl[0] == "NEW"):
                        run += 1
                    elif(sl[0] == "Reward:"):
                        reward = r.float(1)
                        data[run]["reward"].append(reward)
                i += 1;
            return data
    
    # Function to process the logs
    def process_log(self,data):
            
            merged_data_reward = [[ data[i]["reward"][episode] for i in range(len(data))] for episode in range(self.n_eps)]
            merged_data_mean = [np.mean(np.array(aux)) for aux in merged_data_reward]
            merged_data_std = [np.std(np.array(aux)) for aux in merged_data_reward]

            cross_reward = np.mean(merged_data_mean[self.n_eps-100:self.n_eps])

            pro_data = {}
            pro_data["mean reward"] = merged_data_mean
            pro_data['std reward'] = merged_data_std
            pro_data["cross_reward"] = cross_reward
            return pro_data


        


    # Function to get the experimental config and the data from an experiment folder
    def get_data(self,folder):
        
        file_name = gym_perf_folder + folder + "description.json" 
        with open(file_name,'r') as f:
            list_configs = json.load(f)

        list_data = []
        for i in range(len(list_configs)):
            print(i)
            log_file = gym_perf_folder + folder + str(i)+".txt"

            data = self.parse_log(log_file,list_configs[i])
            pro_data = self.process_log(data)
            
            # Append to list of data
            list_data.append(pro_data)

        return list_configs,list_data




