import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# This double convolution happens at each step in the figure
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # the bias is cnacelled by the batch norm
            # the batchnorm from 2016 is not used in the original paper which is from 2015 
            # stride 1, badding 1, dilation 1 produces same size output
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class PNUNET(nn.Module):
    # we input RGB image and output binary classification, 
    #  the features are the number of filters at each layer similar to the paper
    # the smaller the image, the larger the number of filters
    def __init__( self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], ):
        super(PNUNET, self).__init__()
        
               
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()        
                
        #output size = floor( (size-kernel_size)/stride ) + 1
        # in this case will be half size
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                # explained here https://www.youtube.com/watch?v=96_oGE8WyPg
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        #                              512            1024        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # the final 1x1 convolution that just changes the number of channels in the output
        #                               64            1    
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


        

def test():
    x = torch.randn((3, 1, 161, 161))
    model = PNUNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()