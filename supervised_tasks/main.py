# Main file, to run experiments
# Author: Raphael Dang-Nhu
# 28/02/2019

import sys
import os
from utils import *
from optimization_problem import *
from algorithms import *
from project_conf import *
import itertools
from copy import deepcopy

# Class to represent an experiment 
class Experiment:

    def __init__(self,name):
        self.name = name
        return

    # Function to run an experiment
    def run_exp(self,config_index):
            

           config = self.configs[config_index]
           function = config['function']
           mapping = config['mapping']
           algo = config['algo']
           common_config = config['common']
           feedback_config = config['feedback']

           # Printing info
           print("Config:")
           print(json.dumps(config,indent=4))
           
           # Get optimization problem
           problem = OptimizationProblem(function,mapping,feedback_config)
           # Get files
           log_file_name = str(config_index)+".txt"
           debug_file = self.debug_folder+log_file_name
           perf_file = self.perf_folder+log_file_name

           # Run the algorithm
           x = get_algorithm(algo,common_config,mapping,
                   problem,debug_file,perf_file)
           probs = x.solve()

           return
    
    # Unfold list entries in config 
    def unfold(self,entries):

        result = []
        for entry in entries:
            
            # If entry has a list type
            if("type" in entry and entry["type"]=="list"):
                
                keys = [k for k in list(entry.keys()) if k != "type"]
                vals = [entry[key] for key in keys]
                for i in range(len(vals)): 
                    if(not isinstance(vals[i],list)):
                        vals[i] = [vals[i]]
                product = list(itertools.product(*vals))

                # Add every config to product
                for config_aux in product :

                    config = {}

                    for (k,v) in zip(keys,config_aux):
                        config[k] = v

                    result.append(config)

            else:
                result.append(entry)

        return result

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
          
           
           mappings = self.unfold(global_config["mappings"])
           functions = self.unfold(global_config["functions"])
           algos = self.unfold(global_config["algorithms"])
           # Backwards compatibilty for feedbacks
           if("feedbacks" in global_config):
               feedbacks = self.unfold(global_config["feedbacks"])
           else:
               feedbacks = [{ "name":"L1" },] 
            
           # Iterate over mappings
           for (mapping,function,algo,feedback) in list(itertools.product(mappings,functions,algos,feedbacks)):
               # Add file name to map
               description = {}
               # Mapping also depends on function, hence the deepcopy here (for the dimension)
               description['mapping'] = deepcopy(mapping)
               
               # Hack for reinforce problem
               if(algo['name'] == "reinforce"):
                       description['mapping']['output']['n'] = 10
                       description['mapping']['input']['n'] = 10
               description['algo'] = algo
               description['function'] = function
               description['common'] = common_config 
               description['feedback'] = feedback
               self.configs.append(description)

               #config_index += 1


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
def main_func(argv,folder=exp_folder):

   # Read argument
   experiment = Experiment(folder+argv)
   experiment.run()

if __name__ == "__main__":
    main_func(sys.argv[1])
