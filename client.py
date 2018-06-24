import Pyro4
from scipy import misc
import pickle

MeasurementCore = Pyro4.Proxy("PYRONAME:measurement.api")    # use name server object lookup uri shortcut
weight = 90 / 1000. # kg
height = 165 / 100. # meters
image_path = '/media/sparky/Git/king/Documents/PROJECT1/122a2e85-ba98-45e8-afad-a44528c154a6.png'
image = misc.imread(image_path)

measurements, point_cloud, adjusted_weight, adjusted_height = pickle.loads(MeasurementCore.predict(pickle.dumps(image, 0), weight, height))
print measurements
print weight, " : ", adjusted_weight
print height, " : ", adjusted_height