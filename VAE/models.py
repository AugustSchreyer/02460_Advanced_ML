from typing import *
import torch
from torch import nn,Tensor
from torch.distributions import Distribution
from .reparameterize import ReparameterizedDiagonalGaussian


class BaseVAEprob(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, latent_features:int, encoder, decoder, beta=1) -> None:
        super(BaseVAEprob, self).__init__()
        
        self.latent_features = latent_features
        self.beta=beta

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = encoder
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        # mu, sigma        
        self.decoder = decoder
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        z_tmp = self.decoder(z)
        mu, log_sigma =  z_tmp.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}

    def elbo(self, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = self.forward(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = px.log_prob(x).view(pz.size0, -1).sum(dim=1)
        log_pz = pz.log_prob(z).view(pz.size0, -1).sum(dim=1)
        log_qz = qz.log_prob(z).view(qz.size0, -1).sum(dim=1)
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`

        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs

    
class BaseLSTM_VAEprob(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, latent_features:int, encoder, decoder, beta=1) -> None:
        super(BaseLSTM_VAEprob, self).__init__()
        
        self.latent_features = latent_features
        self.beta=beta

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = encoder
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        # mu, sigma        
        self.decoder = decoder
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        z_tmp = self.decoder(z)
        mu, log_sigma =  z_tmp.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}

    def elbo(self, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = self.forward(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = px.log_prob(x.view(-1, int(self.decoder.seq_len))).view(px.size0, -1).sum(dim=1)
        log_pz = pz.log_prob(z).view(pz.size0, -1).sum(dim=1)
        log_qz = qz.log_prob(z).view(qz.size0, -1).sum(dim=1)
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`

        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs

    

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)
from typing import Dict,Any

class Neural_Stat(nn.Module):
    """A Neural Statistician. A Variational Autoencoder with
    * a Gaussian observation model `p_\theta(x | z) = N(x|\mu(z,c),0.1^2)`
    * a Gaussian prior `p(c) = N(c | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, Encoder, Decoder,
                 latent_z:int, z_layers:int, latent_c:int,
                 seq_len, n_features, hidden_dim=50, num_layers=1, bidirectional=True,
                 beta1=1,beta2=1) -> None:
        super(Neural_Stat,self).__init__()
        
        if (z_layers<1) | (latent_z<1) | (latent_c<1):
            print("Number of z layers/latent variables must be greater than 0")
            return

        self.z_layers = z_layers
        self.latent_z = latent_z
        self.latent_c = latent_c
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.beta1 = beta1
        self.beta2 = beta2

        self.encode_c_shape = self.hidden_dim+int(self.bidirectional)*self.hidden_dim
        
        ##############################################
        # Encode h from multiple data sets. Ex 1 data set of 5 data points
        # Dataset -> h_x. Do it 1,2,...,k times.
        ###########################################
        self.encode_h= Encoder
        #####################################################################
        # transform h for each data set (dataset of 5 data points) into 1 c
        # h1,h2....hk -> h_bar -> mu_c, log(sigma_c^2)
        #####################################################################
        self.encode_c = nn.Sequential(
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=self.encode_c_shape, 
                      out_features=self.encode_c_shape),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=self.encode_c_shape, 
                      out_features=self.encode_c_shape),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=self.encode_c_shape, 
                      out_features=2*latent_c,
                      bias =False) # <- note the 2*latent (mean and variance)
        )

        # q_\phi(z_L|x,c) posterior
        #################################################################################
        # torch.cat(c,h) -> z. q(z_L|c,h)
        #################################################################################
        self.encode_z_from_h_c = nn.Sequential(
            nn.Linear(in_features=self.encode_c_shape+latent_c ,
                      out_features=self.encode_c_shape+latent_c),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=self.encode_c_shape+latent_c,
                      out_features=self.encode_c_shape+latent_c),
            nn.ReLU(),
            #nn.Dropout(0.02),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=self.encode_c_shape+latent_c,
                      out_features=2*latent_z,
                      bias =False) # <- note the 2*latent_z
        )

        # q_\phi(z_i|z_{i+1},x,c) posterior
        #################################################################################
        # torch.cat(c,h,z_i) -> z. q(z_i|z_i,c,h)
        #################################################################################
        self.posterior_z_from_z = nn.Sequential(
            nn.Linear(in_features=self.encode_c_shape+latent_c+latent_z , 
                      out_features=self.encode_c_shape+latent_c+latent_z),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=self.encode_c_shape+latent_c+latent_z,
                      out_features=self.encode_c_shape+latent_c+latent_z),
            nn.ReLU(),
            #nn.Dropout(0.02),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=self.encode_c_shape+latent_c+latent_z, 
                      out_features=2*latent_z,
                      bias =False) # <- note the 2*latent_z
        )
        

        ############################3
        # decode c to z, prior (p(z_L|c))
        #########################
        self.decode_c_to_z = nn.Sequential(
            nn.Linear(in_features=latent_c, out_features=50),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
            #nn.Dropout(0.02),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=50, 
                      out_features=2*latent_z,
                      bias =False) # <- note the 2*latent_z
        )        

        ############################3
        # decode c,z_i+1 to z_i, prior (p(z_i|c,z_{i+1}))
        #########################
        self.prior_z_from_z = nn.Sequential(
            nn.Linear(in_features=latent_c+latent_z, out_features=50),
            nn.ReLU(),
            #nn.Dropout(0.02),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
            #nn.Dropout(0.02),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=50, 
                      out_features=2*latent_z,
                      bias =False) # <- note the 2*latent_z
        )        
        
        ####################################
        # p(x|z,c)
        ##################################
        self.decode_x_from_z_c = Decoder
        
        # define the parameters p(c), chosen as p(c) = N(0, I)
        self.register_buffer('p_c_params', torch.zeros(torch.Size([1, 2*latent_c])))
    #################################################


    ###########################################################
    # Prior of c
    ###########################################################
    def prior_c(self, batch_size:int)-> Distribution: 
        """return the distribution `p(c)`"""
        p_c_params = self.p_c_params.expand(batch_size, *self.p_c_params.shape[-1:])
        mu, log_sigma = p_c_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    ##############################################3
    # q(c|D) h1, h2, ... hk -> mu_c , sigma_c, h_all
    ##################################################3
    def statistics_network(self,h_all):
        """ Return the distribution of `p(c|D)`"""
        #print(h_all.shape)
        h_mean = h_all#.mean(dim=1)    
        #print(h_mean.shape)
        c_params = self.encode_c(h_mean)
        mu_c, log_sigma_c = c_params.chunk(2,dim=-1)
        return ReparameterizedDiagonalGaussian(mu_c, log_sigma_c)

    ###################################################
    # Latent decoder network  (prior of z)
    # p(z|c)
    ######################################################
    def latent_decoder_z_prior(self,c,z_L=None,z_i=None):
        """return the distribution `p(z_L|c) = N(z_L | \mu(c), \sigma(c))`"""
        """and the distribution `p(z_i|c,z_{i+1}) = N(z_i | \mu(c,z_{i+1}), \sigma(c,z_{i+1}))`"""
        # Decode from c to z
        p_z_params = self.decode_c_to_z(c)
        # Prior for layer z_L
        mu_L, log_sigma_L = p_z_params.chunk(2, dim=-1)
        prior_z_L = ReparameterizedDiagonalGaussian(mu_L, log_sigma_L)
        # prior for z_i
        if self.z_layers==1:
            return prior_z_L
        else:
            z = torch.cat((z_L,z_i),-1)
            for i in range( (self.z_layers-1)):
                idx = i*self.latent_z
                conc = torch.cat((c,z[:,idx:(idx+self.latent_z)]),-1)
                tmp_par = self.prior_z_from_z(conc)
                tmp_mu, tmp_log_sigma = p_z_params.chunk(2, dim=-1)
                if i==0:
                    mu = tmp_mu
                    log_sigma = tmp_log_sigma
                else:
                    mu = torch.cat((mu,tmp_mu),-1)
                    log_sigma = torch.cat((log_sigma,tmp_log_sigma),-1)
            prior_z_i = ReparameterizedDiagonalGaussian(mu, log_sigma)
            return prior_z_L,prior_z_i
       
            
    
    #####################################3
    # Interference network, posterior
    # q(z|x,c)
    #####################################
    def interference_z_posterior(self,h_all,c_extra):
        """Return z and its posterior distribution of `q(z|x,c) = N(z | \mu(x,c), \sigma(x,c))`"""
        # Concatenate c and h
        conc = torch.cat((c_extra,h_all),dim=-1)
        q_z_params = self.encode_z_from_h_c(conc)
        # Get mu and sigma for z_L
        mu, log_sigma = q_z_params.chunk(2, dim=-1)
        # Temporary distribution for z_L
        tmp_z_dist = ReparameterizedDiagonalGaussian(mu,log_sigma)
        post_z_L = tmp_z_dist
        # Loop for sampling z and dependend layers
        for i in range(self.z_layers):
            # Sample z
            last_z = tmp_z_dist.rsample()
            # concatenate to get 1 z matrix
            #print(z.shape)
            #print(last_z.shape)
            if i==0:
                z_L = last_z
            elif i==1:
                z_i = last_z
            else:
                z_i = torch.cat((z_i,last_z),-1)
            # If it is not the last layer, get new distribution
            if i != (self.z_layers-1):
                # Concatenate for new means/variances
                tmp_conc = torch.cat((conc,last_z),-1)
                z_par = self.posterior_z_from_z(tmp_conc)
                tmp_mu,tmp_log_sig = z_par.chunk(2,dim=-1)
                # temporary distribution for z_i|z_i+1
                tmp_z_dist = ReparameterizedDiagonalGaussian(tmp_mu,tmp_log_sig)
                # Concatenate/save new means/variances
                if i==0:
                    mu_i = tmp_mu
                    log_sig_i = tmp_log_sig
                else:
                    mu_i = torch.cat((mu_i,tmp_mu),-1)
                    log_sig_i = torch.cat((log_sig_i,tmp_log_sig),-1)

        # Save all means and variances as one distribution with extra dimensions
        post_z_i = ReparameterizedDiagonalGaussian(mu_i,log_sig_i)
        # return both z and posterior distribution.
        if self.z_layers==1:
            #z = z_L
            return post_z_L,z_L
        else:
            #z = torch.cat((z_L,z_i))
            return post_z_L,z_L,post_z_i,z_i


   
    def observation_decoder_network(self, z:Tensor,c_extra:Tensor) -> Distribution:
        """return the distribution `p(x|z,c) = N(x | \mu(z,c), \sigma(z ,c))`"""
        conc = torch.cat((c_extra,z),-1)  

        p_x_params = self.decode_x_from_z_c(conc)
        mu_x, log_sigma_x = p_x_params.chunk(2, dim=-1)
        
        #return torch.distributions.Bernoulli(logits=px_logits)
        return ReparameterizedDiagonalGaussian(mu_x, log_sigma_x)
        

    def forward(self,dataset:Tensor) -> Dict[str, Any]:
        """Forward step. See code"""
        batch_size = dataset.size(0)
        # Get from datapoints x to h
        h_all = self.encode_h(dataset)
        # get posterior_c distribution from h_all-> mean
        posterior_c = self.statistics_network(h_all)        
        # prior of c
        prior_c = self.prior_c(batch_size=batch_size)
        # sample c from posterior
        c = posterior_c.rsample()
        n2 = dataset.size(1)
        # Making C-parameters for concatenating
        c_extra = c #torch.cat(n2*[c.reshape((batch_size,1,self.latent_c))],dim=1)
        
        if self.z_layers == 1:
            # getting z and posterior of z
            posterior_z_L,z_L = self.interference_z_posterior(h_all,c_extra)
            z = z_L
            # Prior of z
            prior_z_L = self.latent_decoder_z_prior(c_extra)
            # Distribution of p(x|z,c_extra)
            p_x = self.observation_decoder_network(z,c_extra)
            return {'px': p_x, 
                    'posterior_z_L': posterior_z_L, 
                    'prior_z_L': prior_z_L,'z_L': z_L,
                    'posterior_c': posterior_c, 
                    'prior_c': prior_c,'c': c,
                    'z': z}
        else:
            # getting z and posterior of z
            posterior_z_L,z_L,posterior_z_i,z_i = self.interference_z_posterior(h_all,c_extra)
            # Prior of z
            prior_z_L,prior_z_i = self.latent_decoder_z_prior(c_extra,z_L,z_i)
            z = torch.cat((z_L,z_i),-1)
            # Distribution of p(x|z,c_extra)
            p_x = self.observation_decoder_network(z,c_extra)
            return {'px': p_x, 
                    'posterior_z_L': posterior_z_L,
                    'posterior_z_i': posterior_z_i,
                    'prior_z_L': prior_z_L,'z_L': z_L,
                    'prior_z_i': prior_z_i,'z_i': z_i,
                    'posterior_c': posterior_c, 
                    'prior_c': prior_c,'c': c,
                    'z': z}

    #######################################################
    ################ .elbo ################################
    #######################################################
    def elbo(self,x:Tensor):
        
        # forward pass through the model
        outputs = self.forward(x)
        
        # unpack outputs
        if self.z_layers==1:
            p_x, posterior_z_L, prior_z_L,z_L,\
            posterior_c, prior_c, c, z = [outputs[k] for k in \
                         ['px','posterior_z_L','prior_z_L',
                         'z_L','posterior_c','prior_c','c','z']]
            # evaluate log probabilities
            # summing probabilities
            log_px = reduce(p_x.log_prob(x.squeeze(-1)))
            log_prior_z_L = reduce(prior_z_L.log_prob(z_L))
            log_post_z_L = reduce(posterior_z_L.log_prob(z_L))
            log_prior_c = reduce(prior_c.log_prob(c))
            log_post_c = reduce(posterior_c.log_prob(c))
            # KL and elbo
            kl_z_L = log_post_z_L-log_prior_z_L
            kl_c = log_post_c-log_prior_c
            elbo = log_px - self.beta1*kl_c - self.beta2*kl_z_L
            # Loss function
            loss = -elbo.mean()
            # Saving for diagnostics
            with torch.no_grad():
                diagnostics = {'elbo': elbo.mean(), 'log_px':log_px.mean(), 'kl_z_L': kl_z_L.mean(),'kl_c': kl_c.mean()}

        else:

            p_x, posterior_z_L, posterior_z_i, \
            prior_z_L, z_L, prior_z_i,z_i,\
            posterior_c, prior_c, c, z = [outputs[k] for k in \
                         ['px','posterior_z_L','posterior_z_i',
                          'prior_z_L','z_L','prior_z_i','z_i',
                          'posterior_c','prior_c','c','z']]
            # evaluate log probabilities
            # summing probabilities 
            log_px =reduce(p_x.log_prob(x.squeeze(-1)))
            log_prior_z_L = reduce(prior_z_L.log_prob(z_L))
            log_post_z_L = reduce(posterior_z_L.log_prob(z_L))
            log_prior_z_i = reduce(prior_z_i.log_prob(z_i))
            log_post_z_i = reduce(posterior_z_i.log_prob(z_i))
            
            #post_z = torch.cat((posterior_z_L.log_prob(z_L),posterior_z_i.log_prob(z_i)),2)
            #prior_z = torch.cat((prior_z_L.log_prob(z_L),prior_z_i.log_prob(z_i)),2)
            #log_post_z = reduce(post_z)
            #log_prior_z = reduce(prior_z)

            log_prior_c = reduce(prior_c.log_prob(c))
            log_post_c = reduce(posterior_c.log_prob(c))
            # KL and elbo
            kl_z_L = log_post_z_L-log_prior_z_L
            kl_z_i = log_post_z_i-log_prior_z_i
            #kl_z = log_post_z-log_prior_z
            #print(kl_z_L.mean()+kl_z_i.mean(),kl_z.mean())
            kl_c = log_post_c-log_prior_c
            elbo = log_px-self.beta1*kl_c - self.beta2*kl_z_i-self.beta2*kl_z_L
            # Loss function
            loss = -elbo.mean()
            # Saving for diagnostics
            with torch.no_grad():
                diagnostics = {'elbo': elbo.mean(), 'log_px':log_px.mean(), 'kl_z_L': kl_z_L.mean(),'kl_z_i': kl_z_i.mean(),'kl_c': kl_c.mean()}
        # Return loss, diagnostics and output
        return loss,diagnostics,outputs

    def reconstruction(self,dataset):

        batch_size = dataset.size(0)
        # Get from datapoints x to h
        h_all = self.encode_h(dataset)
        # get posterior_c distribution from h_all-> mean
        posterior_c = self.statistics_network(h_all)        
        c = posterior_c.mu
        # Making C-parameters for concatenating
        n2 = dataset.size(1)
        c_extra = torch.cat(n2*[c.reshape((batch_size,1,self.latent_c))],dim=-1)
        
        if self.z_layers == 1:
            # getting z and posterior of z
            posterior_z_L,z_L = self.interference_z_posterior(h_all,c_extra)
            z = z_L
          
        else:
            # getting z and posterior of z
            posterior_z_L,z_L,posterior_z_i,z_i = self.interference_z_posterior(h_all,c_extra)
            z = torch.cat((z_L,z_i),-1)
        # Distribution of p(x|z,c_extra)
        p_x = self.observation_decoder_network(z,c_extra)
        x = p_x.sample()
        return x