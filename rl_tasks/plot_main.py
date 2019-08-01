# Main plot file for gym
# Author: Raphael Dang-Nhu
# 01/06/2019


import pprint
import log_parser
import sys
import matplotlib.pyplot as pypl
import numpy as np
import os
import math
from gym_project_conf import *
from collections import defaultdict
import json
import numpy as np

sys.path.insert(0,'/home/raphael/Documents/4A/SemesterProject/baselines')

from baselines.common import plot_util as pu


class PlottingFunctions:
    
    def __init__(self,list_configs,list_datas):
        self.algs = ["ppo2", "ddpg"]
        # Get env
        self.env = list_configs[0]["env"]["name"]
        self.n_eps = list_configs[0]["common"]["eps"]
        self.n_benchmark_runs = 10

    # Function to merge baselines
    def merge_baselines(self,results,radius):
        merged_results = [ [r.monitor.r[i] for r in results] for i in range(self.right) ]
        mean_rew = np.array([np.mean(np.array(a)) for a in merged_results])
        std_rew = np.array([np.std(np.array(a)) for a in merged_results])

        return pu.smooth(mean_rew,radius),pu.smooth(std_rew,radius)

    # Function to get color
    def get_color_alg(self,alg):
        if(alg=="ddpg"):
            return 'r'
        else:
            return 'b'

    # Function to plot baselines
    def plot_baselines(self,baselines_folder,radius):
        
        for alg in self.algs:
            
            log_folder = baselines_folder + alg 
            print("log folder",log_folder)
            results = pu.load_results(log_folder)
            mean_rew,std_rew = self.merge_baselines(results,radius)

            color = self.get_color_alg(alg)
            pypl.plot(mean_rew,label=alg,color=color)
            pypl.fill_between(range(self.right),mean_rew-std_rew,mean_rew+std_rew,alpha=0.2,color=color)

    # Fucntion to parse benchamrk log file
    def parse_file(self,file_name):

        data = {"reward":{}}
        with open(file_name,'r') as f:
           lines = f.readlines()
           for line in lines:
               split = line.split()

               if(len(split)>9):
                   episode = int(split[3])
                   if(episode < 200):
                       reward = float(split[9])
                       data["reward"][episode] = reward
        return data

    # Function to merge data
    def merge_data(self,data):

        merged_data_reward = [[ data[i]["reward"][episode] for i in range(len(data))] for episode in range(self.right)]
        merged_data_mean = [np.mean(np.array(aux)) for aux in merged_data_reward]
        merged_data_std = [np.std(np.array(aux)) for aux in merged_data_reward]

        merged_data = {"mean reward":merged_data_mean,'std reward':merged_data_std}
        return merged_data

    # Function to plot the sample efficient baselines
    def plot_sample_eff_baselines(self):
        data = []
        for i in range(self.n_benchmark_runs):
            benchmark_log_file = "logs/MC_logs/"+str(i)+".txt"
            
            data.append(self.parse_file(benchmark_log_file))
        merged_data = self.merge_data(data)
        y = pu.smooth(np.array(merged_data["mean reward"]),self.radius)
        err = pu.smooth(np.array(merged_data["std reward"]),self.radius)
        color='orange'
        pypl.plot(y,label="sample-efficient ddpg",color='orange')
        pypl.fill_between(range(self.right),y-err,y+err,alpha=0.2,color='orange')

def plot_main(folder):

    
    # Add slash to fodler
    if(not folder.endswith("/")):
        folder += "/"


    # Parse the experiment logs
    log_parse = log_parser.LogParser()
    
    # For each configuration, get the data
    list_configs, list_datas = log_parse.get_data(folder)
    plotter = PlottingFunctions(list_configs,list_datas)


    # Get the specific benchmarks
    if(plotter.env == "MountainCarContinuous-v0"):
        radius = 10
        plotter.right = 160
        plotter.radius = radius
        plotter.plot_sample_eff_baselines()
        baselines_folder = "logs/MC_logs/"
        title = "Mountain Car Continuous"
    else:
        baselines_folder = "logs/IP_logs/"
        radius = 70
        plotter.right = 1400
        title = "Inverted Pendulum"
        

    
    # Print the results
    plotter.plot_baselines(baselines_folder,radius)


    # Function to generate a plot with comparison of k in all experimental configs
    algos = {}
    coeffs = {}

    for i in range(len(list_configs)):

        data = list_datas[i]
        config = list_configs[i] 
        
        algo = config["algo"]["name"]
        entry = [config,data]

        if(algo in algos):
            algos[algo].append(entry)
        else:
            algos[algo] = [entry]
        if("coeffs" in config["algo"]):
            coeffs[algo] = config["algo"]["coeffs"]
        #label = self.get_label(config)

    for (algo,entries) in algos.items(): 

        # Find entry with minimum cross error
        maximum = -math.inf
        for entry in entries:
            data = entry[1]

            pprint.pprint(entry[0]["algo"],width=1)
            print("Cross reward",data["cross_reward"],"\n")
            if(data["cross_reward"]>maximum):
                maximum = np.mean(data["cross_reward"])
                min_entry = entry

        pprint.pprint(min_entry[0]["algo"],width=1)
        # Get the test error
        mean_reward = min_entry[1]["mean reward"][:plotter.right]
        std_reward = min_entry[1]["std reward"][:plotter.right]

        # Plot it
        if("legend" in entry[0]["algo"]):
            label = entry[0]["algo"]["legend"]
        else:
            label = entry[0]["algo"]["name"]

        name = entry[0]["algo"]["name"]
        print(name)
    
        x = range(plotter.right)
        y = pu.smooth(np.array(mean_reward),radius)
        err = pu.smooth(np.array(std_reward),radius)
        pypl.plot(y,label="bump",color='g')
        pypl.fill_between(x,y-err,y+err,color='g',alpha=0.2)

    # Do the plot
    pypl.xlabel("Episode")
    pypl.ylabel("reward (average/std of 10 trials)")

    pypl.title(title)
    pypl.xlim(left=0,right=plotter.right)
    pypl.grid()

    if(plotter.env == "MountainCarContinuous-v0"):
        #pypl.legend(loc="middle right")
        pypl.ylim(top = 150)
        pypl.legend(loc=2,prop={'size':7})
        file_name="comparison.png"
    else:
        pypl.legend(loc=2,prop={'size':7})
        #pypl.ylim(bottom = 0)
        file_name="comparison_pendulum.png"

    full_folder = gym_fig_folder+folder
    if not os.path.exists(full_folder):
        os.makedirs(full_folder)

    pypl.savefig(full_folder+file_name)
    pypl.close()


 
# Main plotting function
def main(argv):

    # Avoid overflow error when plotting large number of points
    pypl.rcParams['agg.path.chunksize'] = 10000
    
    plot_main(argv)

main(sys.argv[1])


