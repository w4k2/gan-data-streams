from scipy.ndimage.filters import gaussian_filter1d
from torchvision.utils import save_image
from utils import utils as heatmap_utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")


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

    def save_gan_scores(self, score_lists):
        dcgan_indices = [0, 1, 2]
        wgan_indices = [3, 4, 5]

        for dcgan_idx, wgan_idx in zip(dcgan_indices, wgan_indices):
            x = range(len(score_lists[dcgan_idx]))
            y1 = score_lists[dcgan_idx]
            y2 = score_lists[wgan_idx]

            if dcgan_idx == 0 or dcgan_idx == 1:
                y1_smooth = gaussian_filter1d(y1, sigma=10)
                y2_smooth = gaussian_filter1d(y2, sigma=10)
            else:
                y1_smooth = gaussian_filter1d(y1, sigma=4)
                y2_smooth = gaussian_filter1d(y2, sigma=4)

            plt.plot(x, y1_smooth, 'r')
            plt.plot(x, y2_smooth, 'g')

            plt.xlabel("Chunk")
            plt.ylabel("AUC score")
            plt.show()

    def save_generated_images(self, img_lists):
        i = 0
        for img_list in img_lists:
            for image in img_list:
                filepath = "./data/output/fig" + str(i) + ".png"
                save_image(image, filepath)
                i += 1

    def plot_heatmap(self, relevance_img, img, label, img_index):
        heatmap_utils.heatmap(relevance_img, label, 0.64, 0.64, img, img_index)
