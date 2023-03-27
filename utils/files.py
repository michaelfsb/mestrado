import os

def get_log_file_name(file_name):
    file_name = os.path.basename(file_name).strip()[:-3]
    return 'results/'+file_name+'.txt'

def get_plot_file_name(file_name):
    file_name = os.path.basename(file_name).strip()[:-3]
    return 'results/'+file_name+'.png'

def get_plot_error_file_name(file_name):
    file_name = os.path.basename(file_name).strip()[:-3]
    return 'results/'+file_name+'-error.png'