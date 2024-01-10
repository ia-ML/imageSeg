import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)
        #predictions = (predictions > 0.5).float()  # Apply threshold
        
        # Flatten label and prediction tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice score
        intersection = (predictions * targets).sum()
        dice = (2.*intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # Return Dice loss
        return 1 - dice

    
def test ():
    pass


if __name__ == "__main__":
    test()