import torch
import torch.nn as nn
import torch.nn.optim as optim
import torch.nn.DataParallel as DataParallel
from torchvision import transforms, datasets
from torch.cuda.amp from GradScaler, autocast

def train(model, train_set, val_set, batch_size, num_epochs, lr, criterion, save_dir : str):
    if(torch.cuda.is_available()):
        if(torch.cuda.device_count() > 1):
            model = DataParallel(model)
        else:
            model = model.cuda(0)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, tranforms=data_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, tranforms=data_transform)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            lr, hr = batch
            #em yeu anh hihihihihi
            // train, update


            if((batch_idx + 1) % batch_size == 0):
                // update
                // ssim = 
                // update if ssim improve
            
            
    