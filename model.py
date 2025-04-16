import torch
import torch.nn as nn
from char_map import char_to_idx
nn_classes = len(char_to_idx) + 1
class CRNN(nn.Module):
    def __init__(self, img_height, nn_classes):
        super(CRNN, self).__init__()

        #CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64, kernel_size = 3, stride= 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),

            nn.Conv2d(256,512,kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512,512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),

            nn.Conv2d(512,512, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)) 
        )

        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=False)


        self.fc = nn.Linear(512, nn_classes)

    def forward(self,x):
        x = self.cnn(x)  
        b, c, h, w = x.size()
        assert h == 1, f"Expected height=1 after CNN, got {h}"
        x = x.squeeze(2) 
        x = x.permute(2, 0, 1)  
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x) 
        x = x.permute(1, 0, 2)  
        return x 