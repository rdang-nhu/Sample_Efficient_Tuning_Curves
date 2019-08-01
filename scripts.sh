#!/bin/bash



# Run rl experiment
run_rl(){
	cd rl_tasks
	CUDA_VISIBLE_DEVICES=-1 python3 main_gym.py $1
	cd ..
}

# Plot rl experiments
plot_rl(){
	cd rl_tasks
	python3 plot_main.py $1
	cd ..
}

# Run supervised experiments
run_supervised(){
	cd supervised_tasks
	CUDA_VISIBLE_DEVICES=-1 python3 main.py $1
	cd ..
}

# Plot supervised experiments
plot_supervised(){
	cd supervised_tasks
	python3 plot_main.py $1
	cd ..
}



case $1 in
	("run_rl") run_rl $2;;
	("plot_rl") plot_rl $2;;
	("run_supervised") run_supervised $2;;
	("plot_supervised") plot_supervised $2;;
	
esac

