def h2_flow_converter(output_unit):
    """ Convert hydrogen flow rate from kg/s to Nm3/min and Nl/min"""
    if output_unit == 'Nm3/min':
        return 60*11.126
    elif output_unit == 'Nl/min':
        return 60*11126