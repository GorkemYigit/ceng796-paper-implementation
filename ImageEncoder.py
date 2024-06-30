import torch.nn as nn

# INCOMPLETE / INCORRECT !!
# Left from the time I used an incorrect Image Encoder.
# TODO: Adapt the CRNN to replace ResNet18 !!!

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
         # Modify the first convolutional layer to accept 1 input channel instead of 3
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove last layers to get feature map

    def forward(self, x):
        return self.model(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, num_positions, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_positions, embedding_dim)

    def forward(self, positions):
        return self.embedding(positions)

class ImageConditionExtractor(nn.Module):
    def __init__(self, image_encoder, positional_embedding, attention_pooling):
        super(ImageConditionExtractor, self).__init__()
        self.image_encoder = image_encoder
        self.positional_embedding = positional_embedding
        self.attention_pooling = attention_pooling

    def forward(self, image, positions):
        image_features = self.image_encoder(image)
        # image_features: [width, batch_size, feature_dim]
        image_features = image_features.permute(1, 2, 0)  # [batch_size, feature_dim, width]

        positional_emb = self.positional_embedding(positions)  # Shape: [batch_size, width, feature_dim]
        combined_features = image_features + positional_emb.permute(0, 2, 1)  # [batch_size, feature_dim, width] !!!

        conditional_features = self.attention_pooling(combined_features.permute(0, 2, 1))  # [batch_size, width, feature_dim]
        return conditional_features

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(AttentionPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, x):
        # x: (batch_size, sequence_length, embedding_dim)
        x = x.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, embedding_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)  # Permute back to (batch_size, sequence_length, embedding_dim)


# Example usage
#image_encoder = ImageEncoder()
#positional_embedding = PositionalEmbedding(num_positions=49, embedding_dim=512)
#attention_pooling = AttentionPooling(embedding_dim=512)

#image_condition_extractor = ImageConditionExtractor(image_encoder, positional_embedding, attention_pooling)

#image = torch.randn(1, 3, 224, 224)  # Example input image
#positions = torch.arange(0, 49).unsqueeze(0)  # Example positional ids

#c_i = image_condition_extractor(image, positions)