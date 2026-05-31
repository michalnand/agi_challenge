import torch
import torchvision.models as models


class TestingModel(torch.nn.Module):
    def __init__(self, num_inputs):
        super(TestingModel, self).__init__()

        self.conv0  = torch.nn.Conv2d(num_inputs, 16, kernel_size=3, stride=1, padding=1)
        self.act0   = torch.nn.LeakyReLU()
        self.conv1  = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act1   = torch.nn.LeakyReLU()

        self.head0  = torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.head1  = torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):   
        z = self.conv0(x)   
        z = self.act0(z)
        z = self.conv1(z)
        z = self.act1(z)


        return self.head0(z), self.head1(z)
  




class MobileNetKeypointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load the torchvision model without pretrained weights
        # (In torchvision, weights=None means it's randomly initialized)
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights)

        # 2. Extract ONLY the features (The Encoder)
        # This strips out the global average pooling and the linear classifier
        self.encoder = base_model.features 
        
        # Note: MobileNetV2 features output 1280 channels, and downsample the 
        # input spatial dimensions by a factor of 32 (e.g., 256x256 -> 8x8)
        
        # 3. Simple Decoder to upscale back to the original resolution (32x up)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=4, padding=0), # 4x up
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 64, kernel_size=8, stride=8, padding=0),   # 8x up (4x8 = 32x total)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        # 4. Independent Task Heads
        self.seg_head = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)       # Auxiliary segmentation
        self.keypoint_head = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Blurred keypoints
        
    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        
        # Pass through the torchvision encoder
        features = self.encoder(x) 
        # features shape: (batch_size, 1280, H/32, W/32)
        
        # Upscale back to original image dimensions
        latent_space = self.decoder(features)
        # latent_space shape: (batch_size, 64, H, W)
        
        # Generate raw logits for both tasks
        seg_logits = self.seg_head(latent_space)
        keypoint_logits = self.keypoint_head(latent_space)
        
        return seg_logits, keypoint_logits

# --- Verification ---
if __name__ == "__main__":
    model = MobileNetKeypointModel()

    print(model)
    
    # Simulate a batch of 2 images, 3 channels, 256x256 size
    mock_images = torch.randn(2, 3, 256, 256)
    
    seg_out, kp_out = model(mock_images)
    print("Input shape:             ", mock_images.shape)
    print("Segmentation head shape: ", seg_out.shape)  # Expected: (2, 1, 256, 256)
    print("Keypoints head shape:    ", kp_out.shape)  # Expected: (2, 1, 256, 256)