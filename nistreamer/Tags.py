def analog_output(func):
    func._category = "analog_output"
    return func

def digital_output(func):
    func._category = "digital_output"
    return func

def analog_input(func):
    func._category = "analog_input"
    return func

def digital_input(func):
    func._category = "digital_input"
    return func

def propagate_start_value(func):
    func._propagate_start_value = True
    return func

def propagate_duration(func):
    func._propagate_duration = True
    return func