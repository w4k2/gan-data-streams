from evaluators import DCGANEvaluator, WGANCPEvaluator
from data import DataProvider
from visualizers import DataVisualizer


def run():

    # Parameters
    latent_vector_length = 100
    batch_size = 128
    feature_map_size = 64
    color_channels = 3
    n_gpu = 1
    epochs_per_concept = 1

    data_provider = DataProvider()
    data_visualizer = DataVisualizer()

    dataloader_groups = [
        data_provider.get_celeba_dataloaders(batch_size=batch_size),
        data_provider.get_cifar10_dataloaders(batch_size=batch_size),
        data_provider.get_fashion_mnist_dataloaders(batch_size=batch_size)
    ]

    evaluators = [
        DCGANEvaluator(),
        WGANCPEvaluator()
    ]

    for evaluator in evaluators:
        for dataloaders in dataloader_groups:
            evaluator.initialize(latent_vector_length=latent_vector_length, feature_map_size=feature_map_size,
                                 color_channels=color_channels, n_gpu=n_gpu, data_provider=data_provider,
                                 data_visualizer=data_visualizer)
            evaluator.train(dataloaders=dataloaders, epochs_per_concept=epochs_per_concept)


if __name__ == "__main__":
    run()
