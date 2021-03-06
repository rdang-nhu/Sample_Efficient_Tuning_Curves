This is the source code for the paper **Adaptive Tuning Curve Widths improve Sample Efficient Learning**. The described algorithms are implemented and benchmarked.

## Installing the repository and packages requirements

Clone the repository
 ````bash
 git clone https://github.com/rdang-nhu/Sample_Efficient_Tuning_Curves.git
 ````
Enter the root folder
````bash
 cd Sample_Efficient_Tuning_Curves
 ````
Create a virtual environment and activate it
````bash
virtualenv -p python3 venv; source venv/bin/activate
````
Install the requirements with pip
````bash
pip install -r requirements.txt
````
Now, you should be able to run the experiments.

## Running an experiment

From the root folder, to run a supervised learning experiment, execute
````bash
./scripts.sh run_supervised file.json
````
For a reinforcement learning experiment, execute
````bash
./scripts.sh run_gym file.json
````

## Description of experiments
#### Supervised learning
The files are contained in the **supervised_tasks/experiments/** folder.

1. **benchmark_1D.json** contains the configuration for the one dimensional experiments
2.  **benchmark_reach.json** is for the robotic arm task
3.  **benchmark_throw_ball.json** is for the throw ball task
4.  **comparison_k.json** is for the dynamic vs static width comparison
5.  **incremental_throw_ball.json** is for the incremental learning plots
6.  **plot_synapses.json** is to plot the synapses

#### Reinforcement learning
The files are contained in the **rl_tasks/experiments/** folder.

1. **car.json** is for the MountainCarContinuous environment
2.  **pendulum.json** is for the InvertedPendulum environment

## Experiment logs

Running an experiment file creates a log folder with the same name in **supervised_tasks/logs/perf/** or **rl_tasks/logs/perf** according to the kind of task. In this folder, you will find one file per tested configuration, plus a file **description.json** with the configurations description (used by the plotting engine). Each log file should start with the description of the algorithm used to generate the log.

### Description of log files

A log file starts with the description of the corresponding configuration (algorithm and optimized function). Then, it contains different metrics measured throughout the training, for several runs.

1. *Step* is the current training step.
2. *Error* is the current average error (in case of incremental learning, the samples are split in two disjunct sets. *batch error 1* and *batch error 2* refer to the error on the corresponding set of samples.
3. *Test Error* and *Cross Error* both represent the average error at the end of training, measured on two independent set of samples.
4. In RL experiments, *Episode* refers to the episode number, and *Reward* to the reward for that episode.

### Plotting the results of an experiment

To generate the plots associated with an experiment, add a file **plots.json** to the log folder, and execute
````bash
./scripts.sh plot_supervised <folder>
````
for supervised tasks and 
````bash
./scripts.sh plot_rl <folder>
````
reinforcement learning tasks, where folder is the folder name, such as **folder.json**. This will create a folder in **supervised_tasks/logs/perf/** or **rl_tasks/logs/perf** with the generated figures. The **plots.json** file will define which figures to generate.

#### Content of the plots.json file
1. For all **benchmark** experiments
````json
[
	{ 
		"type":"benchmark"
	}
]
	
````
2. For **comparison_k**
````json
[
	{ 
		"type":"benchmark"
	},
{ 
		"type":"comp k"
	}
]
````
3. For **incremental_throw_ball**
````json
[
	
	{ 
		"type":"incremental error"
	}
]
````
4. For **plot_synapses**
````json
[
	
	{ 
		"type":"incremental error"
	}
]
````
