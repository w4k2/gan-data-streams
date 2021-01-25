from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


class DataVisualizer:

    def plot_generated_figures(self, img_list):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('./visualizers/animation1.mp4', writer=writer)

        plt.show()

    def plot_scores(self, scores):
        x = range(len(scores))
        y = scores

        y_smooth = gaussian_filter1d(y, sigma=4)
        plt.plot(x, y_smooth, 'r')

        plt.xlabel("Chunk")
        plt.ylabel("AUC score")
        plt.show()
