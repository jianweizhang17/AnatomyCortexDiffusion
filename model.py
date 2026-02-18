import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tqdm import tqdm

from layer import *


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



class DN(nn.Module):
    def __init__(self,opt):
        
        super(DN, self).__init__()
        
        self.opt = opt
        
        in_ch = self.opt.input_channel
        out_ch = self.opt.output_channel
        chs = self.opt.chs
        use_att = self.opt.use_att
        attr_ch = self.opt.attr_ch
        start_level = self.opt.start_level
        level_num = len(chs) 
        norm_type = self.opt.norm_type
        dropout_rate = self.opt.dropout_rate
        
        # init condition layers
        self.condition_layers = nn.ParameterDict()
        self.condition_nulls = nn.ParameterDict()
        for condition_name,condition_type in zip(self.opt.attr_condition_names,self.opt.attr_condition_types):
            self.condition_nulls[condition_name] = nn.Parameter(torch.randn(1,attr_ch),requires_grad=False)
            if 'cat' in condition_type: 
                self.condition_layers[condition_name] = ConditionEmbedding(attr_ch,in_type=condition_type.split('_')[0],cat_num=int(condition_type.split('_')[1]))
            elif 'num' == condition_type:
                self.condition_layers[condition_name] = ConditionEmbedding(attr_ch,in_type=condition_type.split('_')[0])
        self.t_emb_layer  = ConditionEmbedding(attr_ch,in_type='num')
        conv_layer = ico_conv_layer

        # init the model...
        conv_param = [start_level]
        self.init_conv = conv_layer(in_ch,chs[0],*conv_param)
        
        self.downs = nn.ModuleList([])
        for i in range(level_num):
            dim_in = chs[i]
            dim_out = chs[i+1] if i < level_num-1 else dim_in
            conv_param = [start_level-i]
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in,dim_in,attr_ch,conv_layer,*conv_param,norm_type=norm_type),
                ResnetBlock(dim_in,dim_in,attr_ch,conv_layer,*conv_param,norm_type=norm_type),
                Attention(dim_in,conv_layer,*conv_param) if use_att[i] else Identity(),
                conv_layer(dim_in,dim_out,*conv_param),
                nn.Dropout(dropout_rate),
                ico_pool_layer(*conv_param) if i < level_num-1 else Identity()
            ]))
        mid_ch = chs[-1]
        conv_param = [start_level-level_num+1]
        self.mid_block1 = ResnetBlock(mid_ch,mid_ch,attr_ch,conv_layer,*conv_param,norm_type=norm_type)
        self.mid_block2 = ResnetBlock(mid_ch,mid_ch,attr_ch,conv_layer,*conv_param,norm_type=norm_type)
      
        self.ups = nn.ModuleList([])
        for i in range(level_num):
            dim_in = chs[level_num-1-i] 
            dim_out = chs[level_num-2-i] if level_num-2-i >= 0 else dim_in
            conv_param = [start_level-level_num+1+i]
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_in*2,dim_out,attr_ch,conv_layer,*conv_param,norm_type=norm_type),
                ResnetBlock(dim_in+dim_out,dim_out,attr_ch,conv_layer,*conv_param,norm_type=norm_type),
                Attention(dim_out,conv_layer,*conv_param) if use_att[level_num-1-i] else Identity(),
                nn.Dropout(dropout_rate),
                ico_upconv_layer() if level_num-2-i >= 0 else Identity()
            ]))
        conv_param = [6]
        self.last_block = ResnetBlock(chs[0],chs[0],attr_ch,conv_layer,*conv_param,norm_type=norm_type)
        self.last_conv = conv_layer(chs[0],out_ch,*conv_param)

    def forward(self,x,t,attr_conditions):
        time_emb = self.t_emb_layer(t)
        
        if len(attr_conditions) == 0:
            for attr_condition in self.condition_nulls:
                time_emb += self.condition_nulls[attr_condition]
        else:
            for attr_condition in attr_conditions:
                time_emb += self.condition_layers[attr_condition](attr_conditions[attr_condition])     
                
        # start forward pass
        x = self.init_conv(x)
        h = []
        for block1,block2,attn,conv,do,downsample in self.downs:
            x = block1(x,time_emb)
            h.append(x)

            x = block2(x,time_emb)
            x = attn(x) + x
            h.append(x)
            
            x = conv(x)
            x = do(x)
            x = downsample(x)
        
        x = self.mid_block1(x,time_emb)
        x = self.mid_block2(x,time_emb)
        
        for block1,block2,attn,do,upsample in self.ups:

            x = torch.cat((x,h.pop()),dim=1)
            x = block1(x,time_emb)

            x = torch.cat((x,h.pop()),dim=1)
            x = block2(x,time_emb)
            x = attn(x) + x
            x = do(x)
            x = upsample(x)

        x = self.last_block(x,time_emb)

        return self.last_conv(x)
    

class GaussianDiffusion(nn.Module):
    def __init__(self,opt):
        super().__init__()
        
        self.opt = opt
        timesteps = self.opt.timesteps
        sampling_timesteps = self.opt.sampling_timesteps
        objective = self.opt.objective
        sec_objective = self.opt.sec_objective
        beta_schedule = self.opt.beta_schedule
        schedule_fn_kwargs = self.opt.schedule_fn_kwargs
        ddim_sampling_eta = self.opt.ddim_sampling_eta
        auto_normalize = self.opt.auto_normalize
        offset_noise_strength = self.opt.offset_noise_strength
        min_snr_loss_weight = self.opt.min_snr_loss_weight
        min_snr_gamma = self.opt.min_snr_gamma
        

        self.model = DN(opt)
        

        # self.channels = self.model.in_ch
        # self.self_condition = self.model.self_condition

        # self.image_size = image_size

        self.objective = objective
        self.sec_objective = sec_objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        
        alphas = 1. - betas
        self.alphas = alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x,t,attr_conditions):
        model_output = self.model(x,t,attr_conditions)
        # print ('model output',model_output.shape)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x,t,attr_conditions):
        preds = self.model_predictions(x,t,attr_conditions)
        x_start = preds[1]

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x,t,attr_conditions,use_cfg=False,cfg_w = 0.1):
        device = self.device; b,*_ = x.shape
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x,batched_times,attr_conditions)

        noise = torch.randn_like(x) if t > 0 else 0. 
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    @torch.inference_mode()
    def sample(self,x,attr_conditions,sample_steps=1000,use_cfg=False,cfg_w = 0.1):
           
        imgs = [x]
        x_starts = []
        
        for t in tqdm(reversed(range(0, sample_steps)), desc = 'sampling loop time step', total = sample_steps):
            img, x_start = self.p_sample(imgs[-1],t,attr_conditions,use_cfg=use_cfg,cfg_w=cfg_w)
            imgs.append(img)
            x_starts.append(x_start)

        return imgs,x_starts
    
    @torch.inference_mode()
    def ddim_sample(self,x,attr_conditions,total_timesteps=1000,sampling_timesteps=50,cfg_w = 0.0):
        
        if cfg_w > 0: print ('using cfg...')
        
        device, eta, objective =  self.device, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = x
        imgs = [img]

        x_start = None
        b,*_ = x.shape
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t = torch.full((b,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img,t,attr_conditions)
            if cfg_w > 0:
                pred_noise_uncond,x_start_uncond,*_ = self.model_predictions(img,t,attr_conditions)
                # pred_noise = (1+cfg_w)*pred_noise - cfg_w*pred_noise_uncond
                pred_noise = pred_noise_uncond + (pred_noise-pred_noise_uncond)*cfg_w

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            pc = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  pc * pred_noise + \
                  sigma * noise

            imgs.append(img)

        return imgs


    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise):

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start,t,attr_conditions):

        noise = torch.randn_like(x_start)

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
    
        pred_target = self.model(x,t,attr_conditions)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        return pred_target,target

    def forward(self, x,attr_conditions):
        device = x.device
        b,*_ = x.shape
        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        t = torch.full((b,),t.item(),device=device).long()
        return self.p_losses(x,t,attr_conditions)
    