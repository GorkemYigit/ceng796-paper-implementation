import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased', embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
        self.model = DistilBertModel.from_pretrained(pretrained_model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, embedding_dim)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        projected_output = self.projection(pooled_output)  
        return projected_output
    
if __name__ == "__main__":
    pass