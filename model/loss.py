import torch

class Perceptual_Loss(torch.nn.Module):
    def __init__(self, weights = [1.0, 1.0, 1,0, 1.0, 1.0]):
        super().__init__()
        self.weights = weights
        self.perceptual_loss = torch.nn.L1Loss()
        
    def forward(self, feature_A, feature_B):
        loss = 0
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.weights)):
            loss += self.weights[i] * self.perceptual_loss(feature_A[i], feature_B[i])
        return loss
    
    """
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = self.vggnet(
            IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = self.vggnet(
            IB_lab, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
    """
class Colorization_Loss(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.rgb_loss = torch.nn.MSELoss()
        self.perceptual_loss = Perceptual_Loss(weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0])
    def forward(self, model_output):
        Img_GT_RGB = model_output['Img_RGB']
        Img_Fine_RGB = model_output['Img_Fine_RGB']
        rgb_loss = self.rgb_loss(Img_GT_RGB, Img_Fine_RGB)
        
        vgg19 = model_output['VGG19']
        feature_A = vgg19(
            Img_GT_RGB.permute(0,3,1,2), ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        feature_B = vgg19(
            Img_Fine_RGB.permute(0,3,1,2), ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        perceptual_Loss = self.perceptual_loss(feature_A, feature_B)
        
        loss_ret = 0.2 * rgb_loss + 0.8 * perceptual_Loss
        return loss_ret
    
    
    