from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation

def plane_plot(x, y):
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

