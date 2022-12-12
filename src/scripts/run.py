import os
import time as t

t0 = t.time()
print("\nRunning make_model_and_data.py")
os.system('python make_model_and_data.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning optimise.py")
os.system('python optimise.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

# t0 = t.time()
# print("\nRunning calc_errors.py")
# os.system('python calc_errors.py')
# print("Done in {:.2f} seconds".format(t.time() - t0))

# t0 = t.time()
# print("\nRunning divergence.py")
# os.system('python divergence.py')
# print("Done in {:.2f} seconds".format(t.time() - t0))