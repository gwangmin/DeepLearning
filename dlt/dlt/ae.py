'''
AutoEncoders
'''

from keras.models import Model
from keras.layers import Lambda, Input, Dense
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def ae_view(models, data, range_=((0,12),(0,12)), scatter_points=10000, scatter_figsize=(12,10), cmap=plt.cm.rainbow):
    '''
    Show AutoEncoder's result

    Args:
        models: tuple. (entire ae, encoder part, decoder part)
        data: tuple. (autoencoder's inputs, labels). Using these data.
        range_: (Optional) range to visualize
        scatter_points: (Optional) Number of point to scatter
        scatter_figsize: (Optional) figsize for scatter
        cmap: (Optional) matplotlib cmap to scatter plot. ex. plt.cm.Blues, plt.cm.Reds, ...
    '''
    # Params interpret
    ae, encoder, decoder = models
    x, y = data
    latent_dim = K.int_shape(decoder.get_input_at(0))[1]
    shape = x[0].shape
    ndim = len(shape)
    if ndim == 3:
        if shape[-1] == 1:
            data_type = 'gray'
        else:
            data_type = 'rgb'
    else:
        data_type = 'not image'

    # Show compare image
    if data_type != 'not image':
        compare_image(ae, data, data_type)

    # Visualize latent space
    if latent_dim == 2:
        scatter_on_latent_space(encoder, data, n=scatter_points, figsize=scatter_figsize, cmap=cmap)
    ae_images(decoder, range_=range_)

def ae_images(decoder, range_=((0,12),(0,12))):
    '''
    Visualize manifold

    Args:
        decoder: decoder part
        range_: (Optional) range to visualize
    '''
    # display a 30x30 2D manifold of the digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(range_[0][0], range_[0][1], n)
    grid_y = np.linspace(range_[1][0], range_[1][1], n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi, yi]])
            x_decoded = decoder.predict(z)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def compare_image(ae, data, image_type):
    '''
    Show and compare
    original, ae output image

    Args:
        ae: entire autoencoder
        data: tuple. (autoencoder inputs, labels)
        image_type: 'gray' or 'rgb'. 'gray' - 1 channel, 'rgb' - 3 channel
    '''
    x,y = data
    for i in [0,2]:
        n = np.random.randint(len(x))
        # Original data
        plt.subplot(2, 2, i+1)
        plt.title('Original ' + str(y[n]))
        if image_type == 'gray':
            plt.imshow(x[n].reshape(28,28))
        elif image_type == 'rgb':
            plt.imshow(x[n])
        # AutoEncoder output
        plt.subplot(2, 2, i+2)
        plt.title('AE')
        if image_type == 'gray':
            plt.imshow(ae.predict_on_batch(x[n:n+1]).reshape(28,28))
        elif image_type == 'rgb':
            plt.imshow(ae.predict_on_batch(x[n:n+1]))
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def scatter_on_latent_space(encoder, data, n=10000, figsize=(12,10), cmap=plt.cm.rainbow):
    '''
    Scatter plot on 2D latent space

    Args:
        encoder: Encoder part
        data: tuple (encoder's input, label).
        n: (Optional) Number of point to scatter
        figsize: (Optional) Figure size
        cmap: (Optional) matplotlib cmap. ex. plt.cm.Blues, plt.cm.Reds, ...
    '''
    # Data
    x_data,y_data = data
    # Size
    plt.figure(figsize=figsize)
    # Scatter
    z = encoder.predict(x_data[:n])
    plt.scatter(z[:, 0], z[:, 1], c=y_data[:n], cmap=cmap)
    # Show
    plt.plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


class SimpleAE:
    '''
    Simple AutoEncoder
    Architecture: Input(input_dim) - Dense - Latent vector(latent_dim) - Dense - Output(input_dim)
    member:
        encoder: Encoder
        decoder: Decoder
        ae: Entire AutoEncoder
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', optimizer='adam', loss='mse'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, optimizer, loss)

    def build_model(self, input_dim, latent_dim, last_activation, optimizer, loss):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build ae
        self.ae = Model(inputs=self.input_x, outputs=self.decoder(self.encoder(self.input_x)))
        self.ae.compile(optimizer=optimizer, loss=loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,))
        encode = Dense(latent_dim, activation='relu')(self.input_x)
        self.encoder = Model(inputs=self.input_x, outputs=encode)

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,))
        decode = Dense(input_dim, activation=last_activation)(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode)


class SimpleDAE:
    '''
    Simple Denoising AutoEncoder
    Architecture: Input(input_dim) - Dense - Latent vector(latent_dim) - Dense - Output(input_dim)
    member:
        encoder: Encoder
        decoder: Decoder
        ae: Entire AutoEncoder
    method:
        add_random_noise(inputs[, mean, std, norm_clip])
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', optimizer='adam', loss='mse'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, optimizer, loss)

    def add_random_noise(self, inputs, mean=0, std=1, norm_clip=False):
        '''
        Add random noise to input
        (from normal dist.)
        '''
        noise = np.random.normal(loc=mean, scale=std, size=inputs.shape)
        result = inputs + noise
        if norm_clip:
            result = np.clip(result, 0., 1.)
        return result

    def build_model(self, input_dim, latent_dim, last_activation, optimizer, loss):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build ae
        self.ae = Model(inputs=self.input_x, outputs=self.decoder(self.encoder(self.input_x)))
        self.ae.compile(optimizer=optimizer, loss=loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,))
        encode = Dense(latent_dim, activation='relu')(self.input_x)
        self.encoder = Model(inputs=self.input_x, outputs=encode)

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,))
        decode = Dense(input_dim, activation=last_activation)(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode)


class SimpleVAE:
    '''
    Simple Variational AutoEncoder
    Architecture: Input - Dense - z_mean    - z(sampled) - Dense - Output
                          Dense - z_log_var
    member:
        encoder: Encoder
        decoder: Decoder
        vae: Entire VAE
    args
        loss: 'mse' or 'bce'
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', loss='', optimizer='adam'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, loss, optimizer)

    def build_model(self, input_dim, latent_dim, last_activation, loss, optimizer):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build entire network
        outputs = self.decoder(self.encoder(self.input_x)[2])
        self.vae = Model(self.input_x, outputs, name='VAE')
        # VAE loss = mse_loss or xent_loss + kl_loss
        self.build_loss(input_dim, loss)
        # Compile
        self.vae.compile(optimizer=optimizer)

    def build_loss(self, input_dim, loss):
        '''
        Build VAE loss (= mse_loss or xent_loss + kl_loss)
        '''
        if loss == 'mse':
            reconstruction_loss = mse(self.input_x, self.vae(self.input_x))
        else:
            reconstruction_loss = binary_crossentropy(self.input_x, self.vae(self.input_x))
        #xent_loss = K.sum(K.binary_crossentropy(self.input_x, self.vae(self.input_x)), axis=-1)
        reconstruction_loss *= input_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,), name='input_x')
        self.z_mean = Dense(latent_dim, name='z_mean')(self.input_x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(self.input_x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(SimpleVAE.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.input_x, [self.z_mean, self.z_log_var, z], name='Encoder')

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,), name='input_latent')
        decode = Dense(input_dim, activation=last_activation, name='decode')(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode, name='Decoder')

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        # K is the keras backend
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

