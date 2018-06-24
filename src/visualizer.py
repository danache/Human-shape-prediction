import matplotlib.pyplot as plt

class Visualizer:

    def show(self, joints0, frame0):
        plt.figure(1)
        plt.clf()

        plt.subplot(121)
        plt.imshow(frame0)


        plt.savefig('image.png')