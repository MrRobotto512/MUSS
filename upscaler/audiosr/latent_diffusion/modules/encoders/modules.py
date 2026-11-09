"""Minimal replacement for AudioSR's encoders/modules.py: only the low-pass audio conditioning
encoder used by super-resolution (the text/CLAP/phoneme encoders and their dependencies are dropped)."""
import torch
import torch.nn as nn

from upscaler.audiosr.latent_diffusion.util import instantiate_from_config


def disabled_train(self, mode=True):
    return self


class VAEFeatureExtract(nn.Module):
    def __init__(self, first_stage_config):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vae = None
        self.instantiate_first_stage(first_stage_config)
        self.device = None
        self.unconditional_cond = None

    def get_unconditional_condition(self, batchsize):
        return self.unconditional_cond.unsqueeze(0).expand(batchsize, -1, -1, -1)

    def instantiate_first_stage(self, config):
        self.vae = instantiate_from_config(config)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.train = disabled_train

    def forward(self, batch):
        assert self.vae.training == False
        if self.device is None:
            self.device = next(self.vae.parameters()).device

        with torch.no_grad():
            vae_embed = self.vae.encode(batch.unsqueeze(1)).sample()

        self.unconditional_cond = -11.4981 + vae_embed[0].clone() * 0.0

        return vae_embed.detach()
