import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import List

class ConditionalVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_class = nn.Conv2d(40, 40, kernel_size=1)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        # in_channels += 1 # To account for the extra label channel
        in_channels = in_channels + 40
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*64, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*64, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 64)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 32,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh()
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # breakpoint()
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 256, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['bevs'].float()
        embedded_class = self.embed_class(y)
        # embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        # breakpoint()
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        # breakpoint()
        # z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]

class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 min_log_sigma: float = -5.,
                 max_log_sigma: float = 5.,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        self.downsample_mu_var = nn.AdaptiveAvgPool2d((1,1))
        # self.downsample_var = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_mu = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Linear(hidden_dims[-1], latent_dim))
        # self.fc_var = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Linear(hidden_dims[-1], latent_dim))
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # self.last_conv = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1), nn.Conv1d(hidden_dims[-1]*64, out_channels=2 * self.latent_dim, kernel_size=1)
        # )


        # Build Decoder
        modules = []
        self.upsample_z = nn.Upsample((8,8))
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 32,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh()
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        result = self.downsample_mu_var(result)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # breakpoint()
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # breakpoint()
        result = result.view(-1, 512, 1, 1)
        result = self.upsample_z(result)
        
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        # clip the log_sigma value for numerical stability
        # log_var = torch.clamp(log_var, self.min_log_sigma, self.max_log_sigma)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def loss_function2(self, P, Q):
        """Compute KL loss between P and Q"""
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)
        kld_loss = F.kl_div(Q.log(), P, reduction='sum')

        return kld_loss

    @staticmethod
    def loss_function3(output):
        
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class SimpleVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 min_log_sigma: float = -5.,
                 max_log_sigma: float = 5.,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        self.downsample_mu_var = nn.AdaptiveAvgPool2d((1,1))
        # self.downsample_var = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_mu = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Linear(hidden_dims[-1], latent_dim))
        # self.fc_var = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Linear(hidden_dims[-1], latent_dim))
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # self.last_conv = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1), nn.Conv1d(hidden_dims[-1]*64, out_channels=2 * self.latent_dim, kernel_size=1)
        # )


        # Build Decoder
        

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        result = self.downsample_mu_var(result)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # breakpoint()
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        # clip the log_sigma value for numerical stability
        log_var = torch.clamp(log_var, self.min_log_sigma, self.max_log_sigma)
        z = self.reparameterize(mu, log_var)

        
        bs, _, h, w = input.shape
        result = z.unsqueeze(-1).unsqueeze(-1).expand(bs, self.latent_dim, h, w)
        return  [result, input, mu, log_var, z]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def loss_function2(self, P, Q):
        """Compute KL loss between P and Q"""
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)
        kld_loss = F.kl_div(Q.log(), P, reduction='sum')

        return kld_loss

    @staticmethod
    def loss_function3(output):
        
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss
class SimpleVAEV2(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 min_log_sigma: float = -5.,
                 max_log_sigma: float = 5.,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(SimpleVAEV2, self).__init__()

        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # average pooling
        # self.downsample_mu_var = nn.AdaptiveAvgPool2d((1,1))

        # 1x1 conv
        self.conv_mu = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)

        # self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # self.last_conv = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1), nn.Conv1d(hidden_dims[-1]*64, out_channels=2 * self.latent_dim, kernel_size=1)
        # )
        

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # breakpoint()
        result = self.encoder(input)
        # breakpoint()
        # result = self.downsample_mu_var(result)
        # result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # breakpoint()
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)

        mu = self.conv_mu(result)
        log_var = self.conv_var(result)

        return [mu, log_var]


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        # clip the log_sigma value for numerical stability
        log_var = torch.clamp(log_var, self.min_log_sigma, self.max_log_sigma)
        z = self.reparameterize(mu, log_var)
        
        # test_tensor=torch.randn([16,32,2,2])

        bs, _, h, w = input.shape
        # breakpoint()
        # result = z.expand(bs, self.latent_dim, 32, 32)
        # result = z.unsqueeze(-1).unsqueeze(-1).expand(bs, self.latent_dim, 1, 1)
        result=z
        return  [result, input, mu, log_var, z]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def loss_function2(self, P, Q):
        """Compute KL loss between P and Q"""
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)
        kld_loss = F.kl_div(Q.log(), P, reduction='sum')

        return kld_loss

    @staticmethod
    def loss_function3(output):
        
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss

class SimpleVAEV3(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 min_log_sigma: float = -5.,
                 max_log_sigma: float = 5.,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(SimpleVAEV3, self).__init__()

        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # average pooling
        # self.downsample_mu_var = nn.AdaptiveAvgPool2d((1,1))

        # 1x1 conv for spatial
        self.conv_mu = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)

        # 1x1 conv for spatial
        self.conv_mu_temp = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim*20, kernel_size=1)
        self.conv_var_temp = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim*20, kernel_size=1)


        # self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # self.last_conv = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1), nn.Conv1d(hidden_dims[-1]*64, out_channels=2 * self.latent_dim, kernel_size=1)
        # )
        

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # breakpoint()
        result = self.encoder(input)
        # breakpoint()
        # result = self.downsample_mu_var(result)
        # result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # breakpoint()
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)

        mu = self.conv_mu(result)
        log_var = self.conv_var(result)
        mu_temp = self.conv_mu_temp(result)
        log_var_temp = self.conv_var_temp(result)

        return [mu, log_var, mu_temp, log_var_temp]


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var, mu_temp, log_var_temp = self.encode(input)
        # clip the log_sigma value for numerical stability
        log_var = torch.clamp(log_var, self.min_log_sigma, self.max_log_sigma)
        z = self.reparameterize(mu, log_var)
        
        # add temp gaussian
        z_temp = self.reparameterize(mu_temp, log_var_temp)
        
        # test_tensor=torch.randn([16,32,2,2])

        bs, _, h, w = input.shape
        # breakpoint()
        # result = z.expand(bs, self.latent_dim, 32, 32)
        # result = z.unsqueeze(-1).unsqueeze(-1).expand(bs, self.latent_dim, 1, 1)
        result=z
        result_temp = z_temp
        return  [result, result_temp, input, mu, log_var, mu_temp, log_var_temp, z, z_temp]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def loss_function2(self, P, Q):
        """Compute KL loss between P and Q"""
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)
        kld_loss = F.kl_div(Q.log(), P, reduction='sum')

        return kld_loss

    @staticmethod
    def loss_function3(output):
        
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss

if __name__ == '__main__':
    cvae = ConditionalVAE(in_channels=32, num_classes=10, latent_dim=50, hidden_dims=[32, 64, 128, 256, 256],
                          bev_size=256)    
    bevs = torch.randn(16, 32, 256, 256)
    # labels = torch.randint(0, 10, (1, 10))
    labels = torch.randn(16, 40, 256, 256)
    out = cvae(bevs, bevs=labels)
    print(out[-1].shape)
    print(cvae.loss_function(*out, M_N=1))

    # present_vae = BetaVAE(in_channels=32, latent_dim=50, hidden_dims=[32, 64, 128, 256, 512], 
    #                       bev_size=256)
    # pre_future_vae = BetaVAE(in_channels=72, latent_dim=50, hidden_dims=[32, 64, 128, 256, 512], 
    #                       bev_size=256)

    present_vae = SimpleVAE(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256, 512], 
                          bev_size=256)
    pre_future_vae = SimpleVAE(in_channels=72, latent_dim=32, hidden_dims=[32, 64, 128, 256, 512], 
                          bev_size=256)

    pre_future_feat = torch.cat([bevs, labels], dim=1)
    pre_out = present_vae(bevs)
    pre_future_out = pre_future_vae(pre_future_feat)

    print(pre_out[-1].shape)
    print(pre_future_out[-1].shape)
    
    Q = pre_out[-1]
    P = pre_future_out[-1]

    print(present_vae.loss_function2(Q, P))

    output_all = dict()
    output_all['present_mu'] = pre_out[-3]
    output_all['present_log_sigma'] = pre_out[-2]
    output_all['future_mu'] = pre_future_out[-3]
    output_all['future_log_sigma'] = pre_future_out[-2]

    print(present_vae.loss_function3(output_all))

