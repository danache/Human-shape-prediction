import Pyro4
from scipy import misc
import pickle

MeasurementCore = Pyro4.Proxy("PYRONAME:measurement.api")    # use name server object lookup uri shortcut
weight = 88 / 1000. # kg
height = 168 / 100. # meters
image_path = '/media/sparky/Git/king/Documents/PROJECT1/722f9502-ffc4-4cb9-99ce-f3e92a7363fb'
image = misc.imread(image_path)

measurements, point_cloud, adjusted_weight, adjusted_height = pickle.loads(MeasurementCore.predict(pickle.dumps(image, 0), weight, height))
print measurements
print weight, " : ", adjusted_weight
print height, " : ", adjusted_height