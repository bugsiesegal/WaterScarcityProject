import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import dataprocessing

data = dataprocessing.load_data()

print(len(data))

d_block = dataprocessing.make_block(data)

print(d_block[0][0].shape)

fig, axs = plt.subplots(2)

pos1 = axs[0].imshow(data[0][0], vmin=-0.25, vmax=0.25)

plt.colorbar(pos1, ax=axs[0])

pos2 = axs[1].imshow(data[-1][0], vmin=-0.25, vmax=0.25)

plt.colorbar(pos2, ax=axs[1])

plt.show()
