from evaluators import GANTrainer
from data import DataProvider
from visualizers import DataVisualizer


def run():

    # Parameters
    latent_vector_length = 100
    feature_map_size = 64
    color_channels = 3
    n_gpu = 1
    num_epochs = 5

    data_provider = DataProvider()
    data_visualizer = DataVisualizer()
    gan_trainer = GANTrainer(latent_vector_length=latent_vector_length, feature_map_size=feature_map_size,
                             color_channels=color_channels, n_gpu=n_gpu)

    dataloader = data_provider.get_celeba_dataloader()
    img_list = gan_trainer.train(dataloader=dataloader, num_epochs=num_epochs)
    data_visualizer.plot_generated_figures(img_list=img_list)


if __name__ == "__main__":
    run()
