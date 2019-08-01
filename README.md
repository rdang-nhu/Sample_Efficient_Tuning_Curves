# Organization of the repository

The **simple_functions** folder contains the code used to generate the plots for supervised learning use cases.
The **gym_test** folder contains the first experiments with reinforcement learning. 

# How to generate an experiment ?

To specify an experiment, go in the **simple_functions/experiments/** folder. It contains a list of json files that describe an experiment. 
To run the associated experiment, just go to the main folder and type (replace throw_ball.json with any json file in the folder)

```bash
./scripts.sh run_sf throw_ball.json
```

It runs the experiments and registers the logs in **simple_functions/logs/perf/**, in a subfolder that is taken from the ["common"]["name"] field of the json file.
Once the experiment is run, you can create a file **plots.json**  in this subfolder that specifies what you would like to plot for these logs.
Once this is done, you can run 

```bash
./scripts.sh run_sf throw_ball
``` 

where **throw_ball** is the name of the subfolder already mentioned. It plots all the associated figures in a subfolder of **simple_functions/figs**

# How is an experiment specified ?

The global syntax is 

```json

{
	"common":{
	    
	},
	"functions":
	[
		{ 
		},
		{
		}
	],
	"algorithms":
	[
		{
		},
		{
		}
	],
	"feedbacks":
	[
		{
		},
		{
		}
	],
	"mappings":
	[
		{
		},
		{
		}
	]
}
```
Have a look at throw_ball.json for a full example.

## Specification of the common parameters

The parameters
1. "name": name of the experiment, used for subfolders
2. "runs": number of runs for each trial
3. "M:" number of training steps for each trial

## Specification of a function 

A function object has the following fields
1. "name": is the name of the function
2. "bounds": has the form [[min_input,max_input],[min_output,max_output]]

A function with multidimensional output has the following form
```json
{    
	"name": "multi dimensional",
	"components":
	[
		{
			"name":"identity",
			"bounds":[[],[0,1]]
		},
		{
			"name":"identity",
			"bounds":[[],[0,1]]
		}
	],
	"bounds": [[0,1],[]]
},
```

## Specification of an algorithm

An algorithm object has at least the field "name". It can have several other fields depending on the algorithm, to set the meta-parameters.
We describe the supported algorithms in a following sections.

## Specification of a feedback

C.f. **throw_ball_feedback.json**.

## Specification of a mapping

A mapping object has the following form
```json
"input": {
},
"output": {
}
```
Each input and output mapping at least has two fields
1. "n": number of neurons
2. "type": name of the mapping


and potentially others particular fields depending on the type.

# How are plots specified ?

A plots.json file is a list of objects, which all have at least a "type" field to specify the plot, and potentially other fields.
```json
[
	{ 
		
	},
	{
	    
	}
]
```

# List of supported functionalities

## Algorithms 

### Algorithm with adaptive k

The algorithm with adaptive k is implemented both for one dimensional input and both one dimensional and multidimensional outputs. 
The specification fo the algorithm has the following form

```json
{
			"name": "adaptiveK",
			"alpha": 0.1,
			"alpha synapse":0.1,
			"counter threshold":10,
			"conf coeff":2
			"error coeff":1.5
}
```
Meaning of the parameters:
1. alpha: exponential decay coefficient for the mean error for each neuron
2. alpha synapse: exponential decay coefficient for the mean error for each neuron (irrelevant for 1D output)
3. counter threshold: number of steps between two potential synapse deletions (irrelevant for 1D output)
4. conf coeff: Ulysse's thesis 5 coefficient
5. error coeff: the synapse is deleted iff average_error_synapse > error coeff * average_error_neuron 
6. mode: "exponential" or "average". apha synapse is irrelevant in the "average" mode


## Functions

To generate an experiment for the ball throwing problem, please have a look at one of the throw_ball.json files for the configuration of the function and of the mappings.

