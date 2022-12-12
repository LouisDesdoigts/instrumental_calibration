import os
import time as t

t0 = t.time()
print("\nRunning plot_progress.py")
os.system('python plot_progress.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_astro_params.py")
os.system('python plot_astro_params.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_aberrations.py")
os.system('python plot_aberrations.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_data_resid.py")
os.system('python plot_data_resid.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_FF.py")
os.system('python plot_FF.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_optics.py")
os.system('python plot_optics.py')
print("Done in {:.2f} seconds".format(t.time() - t0))

t0 = t.time()
print("\nRunning plot_divergence.py")
os.system('python plot_divergence.py')
print("Done in {:.2f} seconds".format(t.time() - t0))