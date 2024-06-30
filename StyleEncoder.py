import torch.nn as nn

class StyleConditionEncoder(nn.Module):
    def __init__(self, num_writers, embedding_dim):
        super(StyleConditionEncoder, self).__init__()
        self.embedding = nn.Embedding(num_writers, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, writer_ids):
        writer_embeds = self.embedding(writer_ids)  # Encode the writer IDs
        projected_embeds = self.projection(writer_embeds)  # Project the embeddings
        return projected_embeds