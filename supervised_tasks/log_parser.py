# Functions to parse the logs
# Author: Raphael Dang-Nhu
# 09/04/2018

from ast import literal_eval as make_tuple
from project_conf import *
import json
import numpy as np
from collections import defaultdict

class LogParser:

    def __init__(self):
        self.window = 500
        return
    
    def convolve(self,array):
        return np.convolve(array,np.ones((self.window,))/self.window,'valid')

    # Function to split errors and k in two 
    def split(self,measurements,input_neurons):
        length = len(measurements)
        right = [measurements[i] for i in range(length) if input_neurons[i] >= 500]
        left = [measurements[i] for i in range(length) if input_neurons[i] < 500]
        return [left,right]

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

        # Config
        self.config = config
        self.common_config = config['common']
        self.n_runs = self.common_config['runs']
        print(self.n_runs)
        self.M = self.common_config['M']

        if('dimension' in self.config['mapping']['output']):
            self.dim = self.config['mapping']['output']['dimension']
        else:
            print("No output dimension specified, taking 2")
            self.dim = 2
        
        # Create dict to populate with data
        data = {}

        # Open files
        with open(file_name,"r") as f:
            
            lines = f.readlines()
            
            # Gathered info
            errors = np.zeros(shape=(self.n_runs,self.M))
            av_error = np.zeros(shape=(self.n_runs,self.M))
            k = np.zeros(shape=(self.dim,self.n_runs,self.M))
            data["input_neurons"] = np.zeros(shape=(self.n_runs,self.M))
            data["cross_error"] = np.zeros(shape=(self.n_runs,),dtype=float)
            data["test_error"] = np.zeros(shape=(self.n_runs,),dtype=float)
            data["batch_error"] = [{} for _ in range(self.n_runs)] 
            data["batch_error_1"] = [{} for _ in range(self.n_runs)] 
            data["batch_error_2"] = [{} for _ in range(self.n_runs)] 
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
                    elif(sl[0] == "Step:"):
                        step = r.int(1)
                    elif(sl[0] == "Cross"):
                        data["cross_error"][run] = r.float(2)
                    elif(sl[0] == "Batch"):
                        aux = sl[1].split(":")
                        if(sl[1] == "Error:"):
                            data["batch_error"][run][batch_step] = r.float(2)
                        elif(sl[1] == "Error1:"):
                            data["batch_error_1"][run][batch_step] = r.float(2)
                        elif(sl[1] == "Error2:"):
                            data["batch_error_2"][run][batch_step] = r.float(2)
                        else:
                            batch_step = int(sl[1].split(":")[1].rstrip())

                    elif(sl[0] == "Test"):
                        data["test_error"][run] = r.float(2)
                    elif(sl[0] == "Error:"):
                        errors[run][step] = r.float(1)
                    elif(sl[0] == "Average"):
                        av_error[run][step] = r.float(2)
                    elif(sl[0] == "K:" or sl[0] == "k"):
                        j = 0
                        ind = 1
                        while(j < self.dim):
                            test = r.int(ind)
                            if(test >= 0):
                                k[j][run][step] = test
                                j += 1
                            ind += 1
                    elif(sl[0] == "k0:"):
                        k = r.int(1)
                    elif(sl[0] == "Input"):
                        if(sl[1] == "neuron:"):
                            data["input_neurons"][run][step] = r.int(2)
                        #if(sl[1] == "md_neuron:"):
#                            data["input_neurons"][run][step] = r.tuple(2)
                    i += 1

            return data,errors,av_error,k
    
    # Function to get mean batch error
    def mean_std_batch_error(self,data,string):

            aux0 = "batch_error"+string
            aux1 = "mean_batch_error"+string
            aux2 = "std_batch_error"+string
            data[aux1] = {}
            data[aux2] = {}

            
            if(len(data[aux0]) > 0):
                for batch_step in data[aux0][0]:
                    array = np.array([data[aux0][x][batch_step] for x in range(self.n_runs)])
                    data[aux1][batch_step] = np.mean(array)
                    data[aux2][batch_step] = np.std(array)

    # Function to process the logs
    def process_log(self,data,errors,av_error,k):

            # Get the k/error distribution
            data['raveled_error'] = np.ravel(av_error)
            data['raveled_k'] = np.array([np.ravel(k[i]) for i in range(self.dim)])

            self.mean_std_batch_error(data,"")
            self.mean_std_batch_error(data,"_1")
            self.mean_std_batch_error(data,"_2")

            # Mean and standard deviation for k/error plot
            # Groupby
            data['d'] = [defaultdict(list) for _ in range(self.dim)]
            for index in range(len(data['raveled_k'])):
                aux_k = np.array([data['raveled_k'][i][index] for i in range(self.dim)])
                aux_err = data['raveled_error'][index]
                for i in range(self.dim):
                    data['d'][i][aux_k[i]].append(aux_err)
                
            #data['keys'] = [key for key in data['d']]
            #data['means'] = [np.mean(data['d'][key]) for key in data['d']]
            #data['stds'] = [np.std(data['d'][key]) for key in data['d']]

            # If relu function, split the errors
            #if(function == "relu"):
            #[data['errors_left'],data['errors_right']] = self.split(data['errors'][0],data['input_neurons'][0])
            #[data['k_left'],data['k_right']] = self.split(data['k'][0],data['input_neurons'][0])

            # Mean the errors
            data['errors'] = np.array(errors).mean(axis=0)
            data['moving_average'] = self.convolve(data['errors'])
            data['av_error'] = np.array(av_error).mean(axis=0)
            data['k'] = [np.array(k[i]).mean(axis=0) for i in range(self.dim)]
            data['k_mov_average'] = [self.convolve(data['k'][i]) for i in range(self.dim)]
            data['convolution_length'] = len(data['moving_average'])


    # Function to get the experimental config and the data from an experiment folder
    def get_data(self,folder):
        
        file_name = perf_folder + folder + "description.json" 
        with open(file_name,'r') as f:
            list_configs = json.load(f)

        list_data = []
        for i in range(len(list_configs)):
            print(i)
            log_file = perf_folder + folder + str(i)+".txt"

            data, errors,av_error,k = self.parse_log(log_file,list_configs[i])
            self.process_log(data,errors, av_error,k)
            
            # Append to list of data
            list_data.append(data)

        return list_configs,list_data




