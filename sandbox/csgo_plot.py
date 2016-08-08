from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import numpy as np
import pdb
def plot_plane(x, y):
	fig = plt.figure(1)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)

	for direction in ["xzero", "yzero"]:
		ax.axis[direction].set_axisline_style("-|>")
		ax.axis[direction].set_visible(True)

	for direction in ["left", "right", "bottom", "top"]:
		ax.axis[direction].set_visible(False)
	ax.plot(x, y)
	plt.show()


def plot_scatter(x,y):

	#Must create new figure for each plot.
	cm_subsection = np.linspace(0.0, 0.4, 20)
	colors = [ cm.jet(k) for k in cm_subsection]

	#Plot creating a fade away color for each line.. 
	fig= plt.figure()
	fig.suptitle('Hacker M4A1-S')
	plt.xlabel('ViewX')
	plt.ylabel('ViewY')
	#loop here over color
	pdb.set_trace()
	plt.scatter(x=x, y=y) #color=colors[i])
	plt.show()


