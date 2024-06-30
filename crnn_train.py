# Training loop for the CRNN model

import CRNN
import torch
from torch.nn import CTCLoss
import torch.optim as optim
from tqdm import tqdm
import Config

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

imgH = 64  
imgW = 256
nc = 1  # input channels (1 for grayscale)
nclass = len("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ") + 1 #.,;:!?()[]{}<>-_+=") + 1  # Number of classes (+1 for CTC 'blank' label)
nh = 256  # LSTM hidden units 

crnn = CRNN(imgH, nc, nclass, nh).to(device)

def character_mapping(character):
    order = ord(character)
    if ord('a') <= order <= ord('z'):
        return order - ord('a')

    elif ord('A') <= order <= ord('Z'):
        return order - ord('A') + 26

    elif ord('0') <= order <= ord('9'):
        return order - ord('0') + 52

    else:
        return 62

criterion = CTCLoss()
optimizer = optim.AdamW(crnn.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.2)


num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for epoch in range(num_epochs):
    crnn.train()
    running_loss = 0.0
    #for images, texts, writer_ids in tqdm(dataloader, unit="batch"):
    for images, texts, writer_ids in tqdm(cvl_train_loader, unit="batch"):
        try:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            print('Possible problem within the dataset!')
            continue

        # calculating target_lengths
        target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')
        flattened_targets = ''.join(texts)
        flattened_targets = [character_mapping(char) for char in flattened_targets]
        flattened_targets = torch.tensor(flattened_targets, dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.any(target_lengths >= 50):
            print(f"WARNING: target length > input length! Continuing..")
            continue

        # forward pass
        outputs = crnn(images).log_softmax(2)  # [T, N, C]

        # calculating input lengths
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            ctc_loss = criterion(outputs, flattened_targets, input_lengths, target_lengths)
        except Exception as e:
            print(f'Error calculating CTC loss: {e}')
            continue

        torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=2.0)

        optimizer.zero_grad()
        ctc_loss.backward()
        optimizer.step()

        if torch.isnan(images).any() or torch.isnan(outputs).any():# or torch.isnan(ctc_loss).any():
            num_nan_images = torch.isnan(images).sum().item()
            print(f'NaN values found in input images: {num_nan_images}')

            num_nan_outputs = torch.isnan(outputs).sum().item()
            print(outputs)
            print(f'NaN values found in output images: {num_nan_outputs}')

            continue

        running_loss += ctc_loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], CTC Loss: {ctc_loss.item()}, running loss: {running_loss}")

torch.save(crnn.state_dict(), 'crnn-100-epoch-cvl.pth')
print("Training completed.")
