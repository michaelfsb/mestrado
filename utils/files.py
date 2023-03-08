import os

def get_optimization_status(ipopt_log_file):
    optimzation_status = ''
    with open(ipopt_log_file) as file:
        for line in file:
            if line.startswith('EXIT'):
                optimzation_status = line.strip()[5:-1]
    return optimzation_status

def get_log_file_name(file_name):
    file_name = os.path.basename(file_name).strip()[:-3]
    return 'results/'+file_name+'.txt'

def get_plot_file_name(file_name):
    file_name = os.path.basename(file_name).strip()[:-3]
    return 'results/'+file_name+'.png'