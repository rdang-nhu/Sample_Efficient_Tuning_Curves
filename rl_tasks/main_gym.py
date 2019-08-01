# Main file, for gym
# Author: Raphael Dang-Nhu
# 18/05/2019

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_functions'))

import main
from gym_utils import *
from optimization_problem import *
from algorithms import *
from project_conf import *
import itertools
from copy import deepcopy
from gym_project_conf import *

# Class to represent an experiment 
class GymExperiment(main.Experiment):


    # Function to run an experiment
    def run_exp(self,config_index):
            

           config = self.configs[config_index]
           algo = config['algo']
           env = config["env"]
           c_conf = config["common"]

           # Printing info
           print("Algo name: "+algo['name'])
           print("Env name: "+env['name'])
           
           # Get problem 
           # Get files
           log_file_name = str(config_index)+".txt"
           debug_file = self.debug_folder+log_file_name
           perf_file = self.perf_folder+log_file_name

           # Run the algorithm
           policy = get_policy(env["name"],algo,c_conf,debug_file,perf_file)
           policy.train()

           return
    

    # Function to create a list of configs
    def list_configs(self):
        
       self.configs = []

       # Open json file
       with open(self.name) as json_file:

           # Get the config
           global_config = json.load(json_file)
           common_config = global_config['common']
           self.exp_name = common_config['name']

           # Create subfolders for the experiment logs
           self.debug_folder = debug_folder+self.exp_name+"/"
           self.perf_folder = perf_folder+self.exp_name+"/"
           if not os.path.exists(self.debug_folder):
                os.makedirs(self.debug_folder)
           if not os.path.exists(self.perf_folder):
                os.makedirs(self.perf_folder)
            
           # Create map from config to index 
           file_description = []
           #config_index = 0
          
           envs = self.unfold(global_config["envs"])
           algos = self.unfold(global_config["algos"])
            
           # Iterate over mappings
           for (algo,env) in list(itertools.product(algos,envs)):
               # Add file name to map
               description = {}
               # Mapping also depends on function, hence the deepcopy here (for the dimension)
               description['algo'] = algo
               description["env"] = env
               description["common"] = common_config
               self.configs.append(description)


           return

    # Function to run an experiment
    def run(self):

        self.list_configs()

        for i in range(len(self.configs)):

            self.run_exp(i)

        # Write map to file in log subfolder
        experiment_file = "description.json"
        debug_experiment_file = self.debug_folder+experiment_file
        perf_experiment_file = self.perf_folder+experiment_file
        with open(debug_experiment_file,'w') as f:
            json.dump(self.configs,f,indent=4)
        with open(perf_experiment_file,'w') as f:
            json.dump(self.configs,f,indent=4)

        return

# Run experiment specified by command line
def main_func(argv):

   # Read argument
   experiment = GymExperiment(gym_exp_folder+argv)
   experiment.run()

main_func(sys.argv[1])
