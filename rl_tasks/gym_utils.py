# Util functions for gym
# Author: Raphael Dang-Nhu
# Date: 18/05/2019

import deep_policy
import bump_policy

def get_policy(env_name,algo_config,c_conf,debug_file,perf_file):
    
    algo_name = algo_config["name"]
    if(algo_name == "bump"):
        return bump_policy.BumpPolicy(env_name,algo_config,c_conf,debug_file,perf_file)
    elif(algo_name == "deep"):
        return deep_policy.DeepPolicy(env_name,algo_config)
