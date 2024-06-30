import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nltk
from collections import Counter

nltk.download('punkt')
from nltk.tokenize import word_tokenize

class CVLDataset(Dataset):
    def __init__(self, data_dir, transform=None, vocab=None, max_length=100, image_size=(224, 224)):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            vocab (dict, optional): Vocabulary for tokenizing labels.
            max_length (int, optional): Maximum length for tokenized labels.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        self.images = []
        self.labels = []
        self.writer_IDs = []
        self.image_size = image_size

        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".tif"):
                        self.images.append(os.path.join(subdir_path, file))
                        # Extract label from filename
                        label = file.split('-')[-1].split('.')[0]
                        self.labels.append(label)
                        self.writer_IDs.append(subdir)

        self.num_distinct_writer_ID = max([int(writer_id) for writer_id in self.writer_IDs])

        # Tokenizer and vocabulary
        self.tokenizer = word_tokenize
        if vocab is None:
            self.vocab = self.build_vocab(self.labels)
        else:
            self.vocab = vocab

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        writer_id = self.writer_IDs[idx]

        image = Image.open(image_path) # .convert('L')  # L is for grayscale conversion

        if self.transform:
            image = self.transform(image)

        # tokenizing and encoding the label
        tokens = self.tokenizer(label)
        token_indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        # padding/truncation
        if len(token_indices) < self.max_length:
            token_indices += [self.vocab['<pad>']] * (self.max_length - len(token_indices))
        else:
            token_indices = token_indices[:self.max_length]

        text_tokens = torch.tensor(token_indices)
        text_positions = torch.arange(0, self.max_length)
        return image, text_tokens, text_positions, torch.tensor(int(writer_id))

    def build_vocab(self, labels):
        counter = Counter()
        for label in labels:
            tokens = self.tokenizer(label)
            counter.update(tokens)
        vocab = {word: i for i, (word, _) in enumerate(counter.items(), start=2)}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        return vocab

    def get_num_distinct_writers(self):
        return self.num_distinct_writer_ID

transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

data_dir = './cvl-database-1-1/trainset/words'

cvl_train_set = CVLDataset(data_dir, transform=transform)
cvl_train_loader = DataLoader(cvl_train_set, batch_size=64, shuffle=True)


if __name__ == "__main__":
    pass