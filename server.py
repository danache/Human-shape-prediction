import Pyro4
import pickle
import main
from measurement import output_measurements

# python -m Pyro4.naming -n 10.10.4.167

@Pyro4.expose
class MeasurementCore(object):

    def predict(self, pickled_picture, weight, height):
        picture = pickle.loads(pickled_picture)
        viz_image, hmr_verts, adjusted_verts, m_adjusted_verts, adjusted_weight, adjusted_height = main.predict(picture, weight, height)
        measurements = output_measurements(m_adjusted_verts)
        return pickle.dumps((measurements, viz_image, hmr_verts, adjusted_verts, adjusted_weight, adjusted_height), protocol=0)


if __name__ == "__main__":
    daemon = Pyro4.Daemon('10.10.4.167')                # make a Pyro daemon
    ns = Pyro4.locateNS()                  # find the name server
    uri = daemon.register(MeasurementCore)   # register the greeting maker as a Pyro object
    ns.register("measurement.api", uri)   # register the object with a name in the name server

    print("Ready.")
    daemon.requestLoop()                   # start the event loop of the server to wait for calls