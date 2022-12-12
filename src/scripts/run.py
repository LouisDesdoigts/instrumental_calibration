import os
os.system('python make_model_and_data.py')
os.system('python optimise.py')
os.system('python calc_error.py')
os.system('python divergence.py')