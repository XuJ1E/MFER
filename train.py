import argparse

import timm
import torch
from tqdm import tqdm
from torch import nn
from torchvision import transforms, datasets
from models.loss_function import LabelSmoothingCrossEntropy, SmoothFunction
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from MFER import convnext_base


parser = argparse.ArgumentParser(description='FER')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--weight_loss', type=float, default=0.001, help='loss weight of feature loss')
parser.add_argument('--bs', type=int, default=32, help='bs')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--epochs', type=int, default=100, help='number of train epoch')
parser.add_argument('--workers', type=int, default=8, help='Number of cpu data loading works')
args = parser.parse_args()


def main(args):
  model = convnext_base(pretrained=False, num_classes=7, drop_path_rate=0.25)
  loss = LabelSmoothingCrossEntropy().to(self.device)
  lossm = FeatureLoss(feat_dim=1024, num_class=7).to(self.device)

  optimizer = torch.optim.AdamW(model.parameters()+lossm.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)  
  
  train_dataset = datasets.ImageFolder(root='./data/RAFDB/dataset/train',
                                                  transform=transforms.Compose([
                                                      transforms.Resize(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])]))
  val_dataset = datasets.ImageFolder(root='./data/RAFDB/dataset/val',
                                                transform=transforms.Compose([
                                                    transforms.Resize(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                ]))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.bs,
                                             shuffle=True,
                                             num_workers=args.workers)
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=args.bs,
                                          shuffle=False,
                                          num_workers=args.workers)
  best_acc = 0
  for epoch in tqdm(range(1, args.epochs + 1)):
      running_loss = 0.0
      correct_sum = 0
      iter_cnt = 0
      model.train()

      for (imgs, targets) in train_loader:
          iter_cnt += 1
          optimizer.zero_grad()

          imgs = imgs.to(device)
          targets = targets.to(device)
            
          feat = model(imgs)
          lossm = lossm(feat, target)
          for i in range(3):
              pred = msc[i](feat[:, 0:i*256+512])
              loss += loss(pred, target)
              preds += pred

          loss = loss + lossm

          loss.backward()
          optimizer.step()
            
          running_loss += loss
          _, predicts = torch.max(out, 1)
          correct_num = torch.eq(predicts, targets).sum()
          correct_sum += correct_num

      acc = correct_sum.float() / float(train_dataset.__len__())
      running_loss = running_loss/iter_cnt
      tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
      with torch.no_grad():
          running_loss = 0.0
          iter_cnt = 0
          bingo_cnt = 0
          sample_cnt = 0
            
            ## for calculating balanced accuracy
          y_true = []
          y_pred = []

          model.eval()
          for (imgs, targets) in val_loader:
              imgs = imgs.to(device)
              targets = targets.to(device)
                
              feat = model(imgs)
              lossm = lossm(feat, target)
              for i in range(3):
                  pred = msc[i](feat[:, 0:i*256+512])
                  loss += loss(pred, target)
                  preds += pred

              loss = loss + lossm

              running_loss += loss
              iter_cnt+=1
              _, predicts = torch.max(out, 1)
              correct_num  = torch.eq(predicts,targets)
              bingo_cnt += correct_num.sum().cpu()
              sample_cnt += out.size(0)
              
        
          running_loss = running_loss/iter_cnt   
          scheduler.step()

          acc = bingo_cnt.float()/float(sample_cnt)
          acc = np.around(acc.numpy(),4)
          best_acc = max(acc,best_acc)

          tqdm.write("[Epoch %d] Validation accuracy:%.4f. acc:%.4f. Loss:%.3f" % (epoch, acc, acc, running_loss))
          tqdm.write("best_acc:" + str(best_acc))

          if acc > 0.92 and acc == best_acc:
              torch.save({'iter': epoch,
                          'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                          os.path.join('checkpoints', "rafdb_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(best_acc)+".pth"))
              tqdm.write('Model saved.')

        
if __name__ == "__main__":        
    main(args)
