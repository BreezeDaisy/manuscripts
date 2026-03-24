import torch
import torch.nn as nn

class MultimodalDecoder(nn.Module):
    def __init__(self, embed_dim, future_steps, k=6):
        super().__init__()
        self.k = k
        self.future_steps = future_steps
        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, k * future_steps * 2)
        )

        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k)
        )

    def forward(self, x_agent, x_encoder, key_padding_mask, num_agents):
        batch_size = x_agent.shape[0]

        pred = self.decoder(x_agent)
        pred = pred.view(batch_size, self.k, self.future_steps, 2)

        pi = self.pi(x_agent)
        pi = torch.softmax(pi, dim=-1)

        return pred, pi