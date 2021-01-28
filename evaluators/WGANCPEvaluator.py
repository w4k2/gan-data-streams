from models.WGANCP import Generator, Discriminator
from detectors import DSDM
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torchvision import utils
import torchvision.models as models
import torch.nn as nn
import torch


class WGANCPEvaluator:
    def __init__(self):

        self._latent_vector_length = None
        self._device = None
        self._data_provider = None
        self._data_visualizer = None

        self._generator = None
        self._discriminator = None

        # Classifiers for calculating GAN quality index
        self._clf_real = None
        self._clf_induced = None
        self._clf_real_linear_layer = None
        self._clf_real_softmax_layer = None
        self._clf_induced_linear_layer = None
        self._clf_induced_softmax_layer = None
        self._clf_real_criterion = None
        self._clf_induced_criterion = None

        self._fixed_noise = None

        # WGAN values from paper
        self._learning_rate = None
        self._beta1 = None
        self.weight_cliping_limit = None

        # Optimizers
        self._discriminator_optimizer = None
        self._generator_optimizer = None

        self._clf_real_optimizer = None
        self._clf_induced_optimizer = None

        self._critic_iter = None

        self._clf_induced_auc = None
        self._auc_scores = None

        self._dsdm = None

    def initialize(self, latent_vector_length=100, feature_map_size=64, color_channels=3, n_gpu=0, data_provider=None,
                   data_visualizer=None):
        self._latent_vector_length = latent_vector_length
        self._device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
        self._data_provider = data_provider
        self._data_visualizer = data_visualizer

        self._generator = Generator(latent_vector_length=latent_vector_length, feature_map_size=feature_map_size,
                                    color_channels=color_channels).to(self._device)
        self._discriminator = Discriminator(feature_map_size=feature_map_size,
                                            color_channels=color_channels).to(self._device)

        # Classifiers for calculating GAN quality index
        self._clf_real = models.resnet18(pretrained=False).to(self._device)
        self._clf_induced = models.resnet18(pretrained=False).to(self._device)
        self._clf_real_linear_layer = nn.Linear(1000, 2).to(self._device)
        self._clf_real_softmax_layer = nn.LogSoftmax(dim=1)
        self._clf_induced_linear_layer = nn.Linear(1000, 2).to(self._device)
        self._clf_induced_softmax_layer = nn.LogSoftmax(dim=1)
        self._clf_real_criterion = nn.NLLLoss()
        self._clf_induced_criterion = nn.NLLLoss()

        if (self._device.type == 'cuda') and (n_gpu > 1):
            self._generator = nn.DataParallel(self._generator, list(range(n_gpu)))
            self._discriminator = nn.DataParallel(self._discriminator, list(range(n_gpu)))
            self._clf_real = nn.DataParallel(self._clf_real, list(range(n_gpu)))
            self._clf_induced = nn.DataParallel(self._clf_induced, list(range(n_gpu)))

        self._fixed_noise = None

        # WGAN values from paper
        self._learning_rate = 0.00005

        self._beta1 = 0.5
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self._discriminator_optimizer = torch.optim.RMSprop(self._discriminator.parameters(), lr=self._learning_rate)
        self._generator_optimizer = torch.optim.RMSprop(self._generator.parameters(), lr=self._learning_rate)

        self._clf_real_optimizer = \
            torch.optim.Adam(self._clf_real.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self._clf_induced_optimizer = \
            torch.optim.Adam(self._clf_induced.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))

        self._critic_iter = 5

        self._clf_induced_auc = None
        self._auc_scores = []

        self._dsdm = DSDM(drift_detection_level=3.0)

    def train(self, dataloaders, epochs_per_concept=5):

        img_list = []
        iteration = 0

        one = torch.FloatTensor([1])
        mone = one * -1

        one = one.to(self._device)
        mone = mone.to(self._device)

        concept = 0
        for dataloader in dataloaders:
            concept += 1
            print("Training dataloader for concept: ", concept)
            for epoch in range(epochs_per_concept):
                print("Epoch: ", epoch)
                # For each batch in the dataloader
                for i, data in enumerate(dataloader, 0):

                    real_images = data[0].to(self._device)
                    real_labels = data[1].to(self._device)
                    batch_size = real_images.size(0)

                    # Requires grad, Generator requires_grad = False
                    for p in self._discriminator.parameters():
                        p.requires_grad = True

                    # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1
                    # Generator forward-loss-backward-update
                    for d_iter in range(self._critic_iter):
                        self._discriminator.zero_grad()

                        # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                        for p in self._discriminator.parameters():
                            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                        noise = torch.rand((batch_size, 100, 1, 1)).to(self._device)
                        real_images, noise = Variable(real_images), Variable(noise)

                        # Train discriminator
                        # WGAN - Training discriminator more iterations than generator
                        # Train with real images
                        discriminator_real_loss = self._discriminator(real_images)
                        discriminator_real_loss = discriminator_real_loss.mean(0).view(1)
                        discriminator_real_loss.backward(one)

                        # Train with fake images
                        noise = Variable(torch.randn(batch_size, 100, 1, 1)).to(self._device)
                        fake_images = self._generator(noise)
                        discriminator_fake_loss = self._discriminator(fake_images)
                        discriminator_fake_loss = discriminator_fake_loss.mean(0).view(1)
                        discriminator_fake_loss.backward(mone)

                        discriminator_loss = discriminator_fake_loss - discriminator_real_loss
                        wasserstein_distance = discriminator_real_loss - discriminator_fake_loss
                        self._discriminator_optimizer.step()

                    # Generator update
                    for p in self._discriminator.parameters():
                        p.requires_grad = False  # to avoid computation

                    self._generator.zero_grad()

                    # Train generator
                    # Compute loss with fake images
                    noise = Variable(torch.randn(batch_size, 100, 1, 1).to(self._device))
                    fake_images = self._generator(noise)
                    generator_loss = self._discriminator(fake_images)
                    generator_loss = generator_loss.mean().mean(0).view(1)
                    generator_loss.backward(one)
                    generator_cost = -generator_loss
                    self._generator_optimizer.step()

                    # Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]' % (epoch, epochs_per_concept, i, len(dataloader)))

                    self._fixed_noise = torch.randn(batch_size, self._latent_vector_length, 1, 1, device=self._device)

                    # Train real data classifier on a single real batch
                    self.train_clf_real(real_images, real_labels)
                    # Generate batch of images and pass them to train GAN-induced classifier
                    generated_images = self._generator(self._fixed_noise)
                    generated_labels = self._clf_real_softmax_layer(self._clf_real_linear_layer(
                        self._clf_real(generated_images)))
                    generated_labels = torch.max(generated_labels, 1)[1]
                    self.train_clf_gan(generated_images, generated_labels, real_images, real_labels)

                    # print("AUC: ", self._clf_gan_auc)
                    self._dsdm.add_element(1 - self._clf_induced_auc)
                    if self._dsdm.detected_change():
                        print("Drift detected, chunk id: ", iteration)

                    self._auc_scores.append(self._clf_induced_auc)

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iteration % 500 == 0) or ((epoch == epochs_per_concept - 1) and (i == len(dataloader) - 1)):
                        with torch.no_grad():
                            fake = self._generator(self._fixed_noise).detach().cpu()
                        img_list.append(utils.make_grid(fake, padding=2, normalize=True))

                    iteration += 1

        self._data_visualizer.plot_scores(scores=self._auc_scores)
        self._data_visualizer.plot_generated_figures(img_list=img_list)

    def train_clf_real(self, inputs, labels):
        self._clf_real_optimizer.zero_grad()
        outputs = self._clf_real_softmax_layer(self._clf_real_linear_layer(self._clf_real(inputs)))
        loss = self._clf_real_criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        self._clf_real_optimizer.step()

    def train_clf_gan(self, train_inputs, train_labels, test_inputs, test_labels):
        with torch.no_grad():
            outputs = self._clf_induced_softmax_layer(self._clf_induced_linear_layer(self._clf_induced(test_inputs)))
            _, predicted = torch.max(outputs.data, 1)

        self._clf_induced_auc = roc_auc_score(torch.max(test_labels.cpu(), 1)[1], predicted.cpu())

        self._clf_induced_optimizer.zero_grad()
        outputs = self._clf_induced_softmax_layer(self._clf_induced_linear_layer(self._clf_induced(train_inputs)))
        loss = self._clf_induced_criterion(outputs, train_labels)
        loss.backward()
        self._clf_induced_optimizer.step()