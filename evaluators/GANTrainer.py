from models import Generator, Discriminator
from torch import nn
from torch import optim
from torchvision import utils
import torch


class GANTrainer:

    def __init__(self, latent_vector_length=100, feature_map_size=64, color_channels=3, n_gpu=0):
        self._latent_vector_length = latent_vector_length

        self._device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

        self._generator = Generator(latent_vector_length=latent_vector_length, feature_map_size=feature_map_size,
                                    color_channels=color_channels, n_gpu=n_gpu, device=self._device).to(self._device)
        self._discriminator = Discriminator(feature_map_size=feature_map_size, color_channels=color_channels,
                                            n_gpu=n_gpu).to(self._device)

        if (self._device.type == 'cuda') and (n_gpu > 1):
            self._generator = nn.DataParallel(self._generator, list(range(n_gpu)))
            self._discriminator = nn.DataParallel(self._discriminator, list(range(n_gpu)))

        self._generator.apply(self.init_weights)
        self._discriminator.apply(self.init_weights)

        # Parameters
        self._learning_rate = 0.0002
        self._beta1 = 0.5

        # Initialize BCELoss function
        self._criterion = nn.NLLLoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self._fixed_noise = None

        # Establish convention for real and fake labels during training
        self._real_label = 1.
        self._fake_label = 0.

        # Setup Adam optimizers for both generator and discriminator
        self._discriminator_optimizer = \
            optim.Adam(self._discriminator.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self._generator_optimizer = \
            optim.Adam(self._generator.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))

    def init_weights(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def train(self, dataloaders, epochs_per_concept=5):
        # Lists to keep track of progress
        img_list = []
        generator_losses = []
        discriminator_losses = []
        iterations = 0

        print("Starting Training Loop...")
        # For each epoch
        for i, dataloader in enumerate(dataloaders):
            print("Training dataloader for concept: ", i)
            for epoch in range(epochs_per_concept):
                print("Epoch: ", epoch)
                # For each batch in the dataloader
                for i, data in enumerate(dataloader, 0):

                    ############################
                    # (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################

                    # Train with all-real batch
                    self._discriminator.zero_grad()

                    # Format batch
                    real_cpu = data[0].to(self._device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), self._real_label, dtype=torch.float, device=self._device).long()

                    # Forward pass real batch through discriminator
                    output = torch.squeeze(self._discriminator(real_cpu))

                    # Calculate loss on all-real batch
                    discriminator_real_error = self._criterion(output, label)

                    # Calculate gradients for discriminator in backward pass
                    discriminator_real_error.backward()
                    D_x = output.mean().item()

                    # Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, self._latent_vector_length, 1, 1, device=self._device)

                    # Generate fake image batch with generator
                    fake = self._generator(noise)
                    label.fill_(self._fake_label)

                    # Classify all fake batch with discriminator
                    output = torch.squeeze(self._discriminator(fake.detach()))

                    # Calculate relevance scores
                    discriminator_pixel_relevance = self._discriminator.get_pixel_layer_relevance()
                    self._generator.calculate_relevance(discriminator_pixel_relevance)

                    # Calculate discriminator's loss on the all-fake batch
                    discriminator_fake_error = self._criterion(output, label)

                    # Calculate the gradients for this batch
                    discriminator_fake_error.backward()
                    D_G_z1 = output.mean().item()

                    # Add the gradients from the all-real and all-fake batches
                    discriminator_error = discriminator_real_error + discriminator_fake_error

                    # Update discriminator
                    self._discriminator_optimizer.step()

                    ############################
                    # (2) Update generator network: maximize log(D(G(z)))
                    ###########################
                    self._generator.zero_grad()
                    label.fill_(self._real_label)  # fake labels are real for generator cost

                    # Since we just updated discriminator, perform another forward pass of all-fake batch through it
                    output = torch.squeeze(self._discriminator(fake))

                    # Calculate generators's loss based on this output
                    generator_error = self._criterion(output, label)

                    # Calculate gradients for G
                    generator_error.backward()
                    D_G_z2 = output.mean().item()

                    # Update G
                    self._generator_optimizer.step()

                    # Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                              % (epoch, epochs_per_concept, i, len(dataloader),
                                 discriminator_error.item(), generator_error.item(), D_x, D_G_z1, D_G_z2))

                    # Save Losses for plotting later
                    generator_losses.append(generator_error.item())
                    discriminator_losses.append(discriminator_error.item())

                    self._fixed_noise = torch.randn(b_size, self._latent_vector_length, 1, 1, device=self._device)

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iterations % 500 == 0) or ((epoch == epochs_per_concept - 1) and (i == len(dataloader) - 1)):
                        with torch.no_grad():
                            fake = self._generator(self._fixed_noise).detach().cpu()
                        img_list.append(utils.make_grid(fake, padding=2, normalize=True))

                    iterations += 1

        return img_list
