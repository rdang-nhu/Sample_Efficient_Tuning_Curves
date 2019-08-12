This is the source code for the paper **Adaptive Tuning Curve Widths improve Sample Efficient Learning**. This code implements 

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


#### Reinforcement learning
The files are contained in the **rl_tasks/experiments/** folder.

