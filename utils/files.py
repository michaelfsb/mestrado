def get_optimization_status(ipopt_log_file):
    optimzation_status = ''
    with open(ipopt_log_file) as file:
        for line in file:
            if line.startswith('EXIT'):
                optimzation_status = line.strip()[5:-1]
    return optimzation_status