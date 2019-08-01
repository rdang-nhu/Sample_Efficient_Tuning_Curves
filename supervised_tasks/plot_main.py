# Plotting functions
# Author: Raphael Dang-Nhu
# 28/02/2019

import sys
import matplotlib.pyplot as pypl
import numpy as np
import os
import math
from project_conf import *
from collections import defaultdict
import log_parser
import plots
import json

    

# Function to read the plots.json file in a folder and generate the corresponding plots
def plot_main(folder):

    # Add slash to fodler
    if(not folder.endswith("/")):
        folder += "/"
        print(folder)

    log_parse = log_parser.LogParser()
    
    # For each configuration, get the data
    list_configs, list_datas = log_parse.get_data(folder)

    plot_env = plots.Plotting_Environment(folder,
            list_configs, list_datas)
    
    # read the plots.json file
    plots_json_file = perf_folder+folder+"plots.json"
    with open(plots_json_file) as plot_json:

        plots_list = json.load(plot_json)

        # Iterate over plots
        i = 0
        for plot in plots_list:

            # Get corresponding plotting function
            function = plot_env.get_plotting_function(plot['type'])

            # Call the function
            function(i)
            i += 1

    return

 
# Main plotting function
def main(argv):

    # Avoid overflow error when plotting large number of points
    pypl.rcParams['agg.path.chunksize'] = 10000
    
    plot_main(argv)

main(sys.argv[1])


