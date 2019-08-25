# Function to generate the plots
# Author : Raphael Dang-Nhu
# Date: 09/04/2018

import pprint
import math
import matplotlib.pyplot as pypl
import numpy as np
import os
import math
from project_conf import *
from collections import defaultdict
import pandas as pd

# Class to generate several plots with same configuration
class Plotting_Environment:

    def __init__(self,folder,list_configs,list_data):
        self.index = 0
        self.folder = fig_folder+folder
        self.window = 500

        # Data and configs
        self.list_configs = list_configs
        self.list_datas = list_data
        self.n_configs = len(self.list_configs)

        # Create folder if it does not exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        return
    
    def convolve(self,array):
        return np.convolve(array,np.ones((self.window,))/self.window,'valid')

    def start_fig(self,index):
        self.fig =  pypl.figure(index)
        return self.fig

    def plot(self,size=4):

        #pypl.legend(prop={'size':size},bbox_to_anchor=(0.55,0.5))
        pypl.legend(prop={'size':size},loc=1)
        pypl.xlim(left=0)
        pypl.ylim(bottom=0)
        pypl.grid(True)
        #pypl.subplots_adjust(bottom=0.3)
        #self.fig.text(.5,0.05,self.text, ha='center')

    def save(self,name):
        pypl.savefig(self.folder+self.title+"_"+name,dpi=1000)

    def get_label(self,config):
        label = ""
        index = 0
        for param in config['algo']:
            if(index > 0 and index % 5 == 0):
                label += "\n"
            label += param +": "+str(config['algo'][param])[0:10]+", "
            index += 1
        return label

    def get_color(self,name):
                if(name=="MultiFloKeepSynapses"):
                    color="C0"
                elif(name.startswith("MultiFloKeep")):
                    color="C"+name[-1]
                elif(name=="perceptron"):
                    color="C1"
                elif(name.startswith("MultiFloTheory")):
                    color="C"+name[-1]
                    color="C0"
                elif(name=="reinforce"):
                    color="C2"
                else:
                    color="C5"
                return color

    # Function to generate a plot with comparison of the errors in all experimental configs
    def group_error_plot(self,index):
        self.start_fig(index)

        for i in range(self.n_configs):
            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            label = self.get_label(config)

            pypl.plot(range(data['convolution_length']),
                    data['moving_average'],label=label)

        self.plot()
        pypl.title("Evolution of Error (Running Average)") 
        pypl.savefig(self.folder+"group_error.png",dpi=2000)
        pypl.close()

    def dict_to_array(self,test_error,test_std):
        x = []
        y = []
        std = []
        for key in sorted(test_error):
    #        if(key % 32 == 0):
                x.append(key)
                y.append(test_error[key])
                std.append(test_std[key])
        return np.array(x),np.array(y),np.array(std)

    # Function to generate a plot with comparison of the errors in all experimental configs
    def group_error_incr(self,index):
        self.start_fig(index)

        for i in range(self.n_configs):
            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            label = self.get_label(config)

            x1,y1,std1 = self.dict_to_array(data["mean_batch_error_1"],data["std_batch_error_1"])
            x2,y2,std2 = self.dict_to_array(data["mean_batch_error_2"],data["std_batch_error_2"])
            x,y,std = self.dict_to_array(data["mean_batch_error"],data["std_batch_error"])

            if(config["algo"]["name"].startswith("Final")):
                color1 = "C0"
                color2 = "C1"
                color = "C2"
                name = "Bump, " 
            else:
                color1 = "C2"
                color2 = "C3"
                name = "NN, "
            
            label1 = "Random short-distance target"
            label2 = "Random long-distance target"
            label3 = "Random target"

            pypl.plot(x1,y1,label=label1,color=color1)
            pypl.fill_between(x1,y1-std1,y1+std1,alpha=0.2,color=color1)
            pypl.plot(x2,y2,label=label2,color=color2)
            pypl.fill_between(x2,y2-std2,y2+std2,alpha=0.2,color=color2)
            pypl.plot(x1,y,label=label3,color=color)
            pypl.fill_between(x1,y-std,y+std,alpha=0.2,color=color)

        pypl.axvline(x=500,linestyle='dashed',color='black')
        pypl.text(250,0.3,'Training on short\ndistance target',size='medium',horizontalalignment='center')
        pypl.text(750,0.3,'Training on long\ndistance target',size='medium',horizontalalignment='center')
        pypl.ylabel("Error")
        pypl.xlabel("Number of training samples")
        pypl.title("Throw ball") 
        self.plot(8)
        pypl.xlim(left=-50)
        pypl.savefig(self.folder+"incremental_error.png",dpi=1000)

    # Function to generate a plot with comparison of the errors in all experimental configs
    def group_new(self,index):
        self.start_fig(index)

        for i in range(self.n_configs):
            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            label = self.get_label(config)

            pypl.plot(range(data['convolution_length']),
                    data['moving_average'],label=label)

        pypl.title("Evolution of Error (Running Average)") 
        self.plot()
        pypl.savefig(self.folder+"group_error.png",dpi=2000)
        

    # Function to generate a plot with comparison of k in all experimental configs
    def group_k_plot(self,start_index):
        
        if('dimension' in self.list_configs[0]['mapping']['output']):
            self.dim = self.list_configs[0]['mapping']['output']['dimension']
        else:
            print("No output dimension specified, taking 2")
            self.dim = 2
        for index in range(self.dim):

            self.start_fig(start_index+index)

            for i in range(self.n_configs):
                data = self.list_datas[i]
                config = self.list_configs[i] 

                label = self.get_label(config)

                pypl.plot(range(data['convolution_length']),
                        data['k_mov_average'][index],label=label)
        
            pypl.title("Evolution of K (Running Average), dimension: "+str(index))
            self.plot()
            pypl.savefig(self.folder+"group_k_dimension_"+str(index)+".png",dpi=2000)

    def get_f_c(self):

        functions = {}
        coeffs = {}

        for i in range(self.n_configs):

            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            algo = config["algo"]["name"]
            function = config["function"]["name"]
            
            entry = [config,data]
            if(function in functions):
                if(algo in functions[function]):
                    functions[function][algo].append(entry)
                else:
                    functions[function][algo] = [entry]
            else:
                    functions[function] = {}
                    functions[function][algo] = [entry]

            if("coeffs" in config["algo"]):
                coeffs[algo] = config["algo"]["coeffs"]
                functions[function].append

            label = self.get_label(config)

        return functions, coeffs
    
    # Function to convert legend
    def convert_legend(self,legend):
        if(legend == "NN"):
            return "MLP, backprop, opt. par."
        if(legend == "NN, one hot encoding"):
            return "MLP one-hot, backprop"
        if(legend == "Reinforce"):
            return "MLP, reinforce"
        if(legend == "Bump"):
            return "Dynamic bump"
        return legend

    # Function to generate a plot with comparison of k in all experimental configs
    def benchmark(self,start_index):
        
        functions,coeffs = self.get_f_c()

        i = 0 
        for function in functions:
            self.start_fig(i)
            i = i+1
            for (algo,entries) in functions[function].items(): 

                # Find entry with minimum cross error
                minimum = math.inf
                for entry in entries:
                    data = entry[1]
                    if(np.mean(data["cross_error"])<minimum):
                        minimum = np.mean(data["cross_error"])
                        min_entry = entry
        
                print("\n")
                print("Function",function)
                pprint.pprint(min_entry[0]["algo"],width=1)
                # Get the test error
                test_error = min_entry[1]["mean_batch_error"]
                test_std = min_entry[1]["std_batch_error"]
                x,y,std = self.dict_to_array(test_error,test_std)
                # Plot it
                if("legend" in entry[0]["algo"]):
                    label = entry[0]["algo"]["legend"]
                else:
                    label = entry[0]["algo"]["name"]
                label = self.convert_legend(label)

                name = entry[0]["algo"]["name"]
                print(name)
                color = self.get_color(name)
                
                test_name = entry[0]["algo"]["name"]
                if(test_name != "AS" and not "one" in test_name):
                    x = x[:36]
                    y = np.array(y[:36])
                    std = np.array(std[:36])
                    pypl.plot(x,y,label=label,color=color)
                    pypl.fill_between(x,y-std,y+std,alpha=0.2,color=color)
            pypl.xlabel("Number of training samples")
            pypl.ylabel("Error")
            self.plot(9)
            if("legend" in entry[0]["function"]):
                title = entry[0]["function"]["legend"]
            else:
                title = entry[0]["function"]["name"]
            pypl.xlim(left=-50)
            title_lower = title[0] + title[1:].lower()
            pypl.title(title_lower)
            title = title.replace(" ","-")
            pypl.savefig(self.folder+title+".png")
            pypl.close()
            #pypl.show()


        return
    

    def get_time(self,x,y,std):

        perf = math.inf
        for i in range(len(x)):

            if(y[i] < perf):
                perf = y[i]

        for i in range(len(x)):

            if(y[i] < 1.5*perf):
                time = i
                return perf,x[time],std[time]


    # Function to generate second comp k plot
    def comp_k(self,start_index):
        
        functions,coeffs = self.get_f_c()

        i = 0 
        for function in functions:
            self.start_fig(i)
            i = i+1
            for (algo,entries) in functions[function].items(): 

                # Find entry with minimum cross error
                minimum = math.inf
                for entry in entries:
                    data = entry[1]
                    if(np.mean(data["cross_error"])<minimum):
                        minimum = np.mean(data["cross_error"])
                        min_entry = entry
        
                print("\n")
                print("Function",function)
                pprint.pprint(min_entry[0]["algo"],width=1)
                # Get the test error
                test_error = min_entry[1]["mean_batch_error"]
                test_std = min_entry[1]["std_batch_error"]
                x,y,std = self.dict_to_array(test_error,test_std)
                # Plot it
                if("legend" in entry[0]["algo"]):
                    label = entry[0]["algo"]["legend"]
                else:
                    label = entry[0]["algo"]["name"]

                name = entry[0]["algo"]["name"]
                print(name)
                color = self.get_color(name)

                    
                final_performance,time,err = self.get_time(x,y,std)
                #pypl.errorbar(time,final_performance,yerr=err,capsize=5,marker='+',color="C0",markersize=15)
                pypl.plot(time,final_performance,marker='+',color="C0",markersize=15)
                if("150" in label):
                    delta = -0.001
                    deltax = -70
                elif("140" in label):
                    delta = -0.002
                    deltax = -70
                elif("130" in label):
                    delta = -0.001
                    deltax = 30
                elif(len(label) < 6):
                    delta = -0.0015
                    deltax = -55
                else: 
                    delta = -0.0015
                    deltax = -70
                pypl.annotate(label[3:],(time+deltax,final_performance+delta))

            pypl.xlabel("Number of training samples to reach final error")
            pypl.ylabel("Final error")
            self.plot(size=12)
            if("legend" in entry[0]["function"]):
                title = entry[0]["function"]["legend"]
            else:
                title = entry[0]["function"]["name"]
            #pypl.title(title)
            title = title.replace(" ","-")
            pypl.xlim(left=-50)
            pypl.savefig(self.folder+"comparison_k_interactive.png")
            pypl.close()
            #pypl.show()


        return

    # Latex chart with test error and std
    def chart_test(self,index):

        for i in range(self.n_configs):

            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            test_error = data["mean_batch_error"]
            test_std = data["std_batch_error"]
            x,y,std = self.dict_to_array(test_error,test_std)
            y = np.around(y,decimals=3)
            std = np.around(std,decimals=3)
            with open(self.folder+"error_chart.txt","w") as f:
                rows = ["Error","Std"]
                columns=np.array(config["algo"]["steps plot"]).astype(int)
                df = pd.DataFrame([y[:15],std[:15]],index=rows,columns=columns[:15])
                f.write(df.to_latex())
                #df2 = pd.DataFrame([y[8:15],std[8:15]],index=rows,columns=columns[8:15])
                #df2.iloc[0].apply(int)
                #f.write(df2.to_latex())



    # Chart with cross and test error
    def chart_cross_test(self,index):

        functions = {}
        coeffs = {}

        for i in range(self.n_configs):


            data = self.list_datas[i]
            config = self.list_configs[i] 
            
            function = config["function"]["name"]
                
            label = self.get_label(config)

            layer_width = config["algo"]["width layers"]
            learn_rate = config["algo"]["learning rate"]

            entry = [layer_width,learn_rate,np.mean(data["cross_error"]),np.mean(data["test_error"])]
            if(function in functions):
                functions[function].append(entry)
            else:
                functions[function] = [entry]
                if("coeffs" in config["function"]):
                    coeffs[function] = config["function"]["coeffs"]
    

        with open(self.folder+"benchmark.txt","w") as f:
            columns = ["Layer Width","Learning Rate","Cross Error","Test Error"]
            for function in functions: 
                df = pd.DataFrame(functions[function],columns=columns)
                df = df.sort_values(by=["Cross Error"])

                f.write("\n")
                f.write(function)
                f.write("\n")
                if(function in coeffs):
                    f.write("Coeffs: ")
                    f.write(str(coeffs[function]))
                    f.write("\n")
                f.write(df.to_string(index=False))


            

    # Function to generate folder name
    def get_folder_name(self,M,alpha,conf_coeff):

        folder = fig_folder_ad_k+str(M)+"_steps/alpha_"+str(alpha)+"/coeff_"+str(conf_coeff)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)


    def error_plotter(self,file_name,i):

        # Open files
        with open(file_name,"r") as f:
            

            moving_average = self.convolve(errors)

            final_error = errors[M:]
            average_final_error = np.mean(final_error)

            # Do the plotting
            fig = pypl.figure(i)
            pypl.axvline(x=M,color='r')
            text = function+" function, input in [" + str(ilb) + ","\
                    + str(irb) + "], output in ["+ str(olb)+","+str(orb)+"],\n"+\
                    str(M) + " algorithm steps, "+\
                    "interval length of " + str(k) + ",\n " +\
                    str(nin) + " input neurons, "+\
                    str(non) + " output neurons, "
            pypl.plot(range(M),moving_average[:M],label="training, running average, "+str(window)+" steps")
            pypl.plot(range(M,len(errors)),final_error,label="after final prob. update")
            pypl.legend()
            pypl.ylabel("Error")
            pypl.xlabel("Step")
            pypl.xlim(xmin=0)
            pypl.ylim(ymin=0)
            pypl.grid(True)
            pypl.subplots_adjust(bottom=0.3)
            fig.text(0.8,0.5,"Average error:\n "+str(average_final_error))
            fig.text(.5,0.05,text, ha='center')
            pypl.savefig(fig_folder+function+"_"+str(i),dpi=1000)


        
    # Function to plot the adaptive k error
    def error_plotter_adaptive_k(self,file_name,fig_index):

            
            # Where to save the images
            plots_folder = self.get_folder_name(M,alpha,conf_coeff)


            # Get the theoretical line for scatter plot
            max_k = max(raveled_k)
            th_k = np.linspace(0,max_k)
            th_th = conf_coeff*th_k*coeff_int / float(non)

            # Get the theoretical line for error plot
            th_step = np.arange(M)
            # nin or non ?
            th_k_ev = non/np.sqrt(th_step)

            # Get the parameters
            title = function+"_adaptiveK_"+str(fig_index)
            text = function+" function, coeffs: "+ coeff_string +\
                    "input in [" + str(ilb) + ","+\
                    str(irb) + "], output in ["+ str(olb)+","+str(orb)+"],\n"+\
                    str(M) + " algorithm steps, "+\
                    "alpha = " + str(alpha) + ", " +\
                    "security coeff = " + str(conf_coeff) + ",\n" +\
                    str(nin) + " input neurons, "+\
                    str(non) + " output neurons"
            plotter = self.Plotter(plots_folder,title,text)

            # Do the plots
            j = 4*fig_index
            plotter.start_fig(j)
            if(function == "relu"):
                left = self.convolve(errors_left)
                right = self.convolve(errors_right)
                pypl.plot(range(len(left)),left,label="Left Error")
                pypl.plot(range(len(right)),right,label="Right Error")
            else: 
                pypl.plot(range(len(moving_average)),moving_average,label="Error")
            #pypl.plot(range(M),av_error,label="Average Error")
            pypl.xlabel("Step")
            plotter.plot()
            plotter.save("error")

            j0 = j+1
            plotter.start_fig(j0)
            if(function == "relu"):
                left = self.convolve(k_left)
                right = self.convolve(k_right)
                pypl.plot(range(len(left)),left,label="Left K")
                pypl.plot(range(len(right)),right,label="Right K")
            else: 
                pypl.plot(range(len(k_mov_average)),k_mov_average,label="K")
                pypl.plot(th_step,th_k_ev,label="neuron_number/sqrt(t)",color='r')
            pypl.xlabel("Step")
            plotter.plot()
            pypl.ylim(ymax=max_k+5)
            plotter.save("k")


            # Do the second plot
            if(function != "relu"):
                j1 = j+2
                plotter.start_fig(j1)
                pypl.scatter(raveled_k,raveled_error,s=2)
                pypl.plot(th_k,th_th,label="target (sec_coeff*lin_coeff*k/neuron_number)",color='r')
                pypl.xlabel("k")
                pypl.ylabel("error threshold")
                plotter.plot()
                plotter.save("scatter")

                # Do the third plot
                j2= j+3
                plotter.start_fig(j2)
                pypl.errorbar(keys,means,stds,marker='o',capsize=2,linestyle='None',markersize=2)
                pypl.plot(th_k,th_th,label="target (5*coeff*k/neuron_number)",color='r')
                pypl.xlabel("k")
                pypl.ylabel("error threshold")
                plotter.plot()
                plotter.save("mean_dev")

    # Function to get the plotting function from name
    def get_plotting_function(self,name):

        if(name == "group error"):
            return self.group_error_plot
        elif(name == "group k"):
            return self.group_k_plot
        elif(name == "chart cross test"):
            return self.chart_cross_test
        elif(name == "benchmark"):
            return self.benchmark
        elif(name == "new_plot"):
            return self.group_new
        elif(name == "incremental error"):
            return self.group_error_incr
        elif(name == "comp k"):
            return self.comp_k
        return getattr(self,name)

    def iterate_logs(self,plotter,folder):

        # Iterate over perf files
        i = 0
        for filename in os.listdir(folder):
            
            item = folder+filename
            if(os.path.isfile(item)):
                plotter(item,i)
                i = i+1
        return folder

