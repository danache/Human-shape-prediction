import Pyro4
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D  # <-- Note the capitalization!
import pickle
from pylab import *



def debug_display_cloud(verts):
    plt.clf()
    fig = plt.figure(1)
    ax3d = Axes3D(fig)
    ax3d.clear()
    ax3d.set_aspect("equal")
    ax3d.set_xlim3d(-1, 1)
    ax3d.set_ylim3d(-1, 1)
    ax3d.set_zlim3d(-1, 1)
    for vert in verts:
        ax3d.plot(vert[:,0], vert[:,1], vert[:,2], ',')
    plt.draw()
    plt.show()


MeasurementCore = Pyro4.Proxy("PYRONAME:measurement.api")    # use name server object lookup uri shortcut
height = 168 / 100. # meters
weight = 88 / 1000.
image_path = '/media/sparky/Git/king/Documents/PROJECT1/722f9502-ffc4-4cb9-99ce-f3e92a7363fb'
image = misc.imread(image_path)


measurements, point_cloud, adjusted_weight, adjusted_height = pickle.loads(MeasurementCore.predict(pickle.dumps(image, 0), weight, height))
print measurements
print weight, " : ", adjusted_weight
print height, " : ", adjusted_height

debug_display_cloud([point_cloud])