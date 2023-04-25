import torch
from torchvision import transforms, datasets
import torchvision.models as models
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# set dataset
dataset = datasets.ImageFolder(root='./data/RAFDB/dataset/val',
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

# set model
# we display the example model as networks
net = models.resnet18(pretrained=True)
net = net.to(device)

def generate_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))
                
    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)
    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('Ploting t-SNE ...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['RAF-DB'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='RAF-DB',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.8
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(save_dir,'tsne.png'), bbox_inches='tight')
    print('Ending ploting ...')
    
    
if __name__ == '__main__':
    save_dir = './checkpoints/tsne/'
    targets, outputs = generate_features()
tsne_plot(save_dir, targets, outputs)
