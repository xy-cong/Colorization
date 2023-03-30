import torch
import torch.nn as nn
from model.VGG19 import VGG19_pytorch
from model.WarpNet import WarpNet
from utils.util import *
from model.unet_model import UNet
from skimage import color
import model.Data as Data


class Colorization(nn.Module):
    def __init__(self, conf, image_size=256):
        super().__init__()
        self.vggnet = VGG19_pytorch()
        self.vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
        self.vggnet.eval()
        for param in self.vggnet.parameters():
            param.requires_grad = False
        self.WarpNet = WarpNet(conf.batch_size)
        self.UNet = UNet()
        self.image_size = image_size
        self.batch_transform = Data.BatchTransform(image_size=(self.image_size, self.image_size))
        # import ipdb; ipdb.set_trace()
        if conf.pre_trained_unet:
            unet_pth = torch.load("data/unet.pth")
            unet_pth['outc.conv.weight'] = torch.randn((3,64,1,1))
            unet_pth['outc.conv.bias'] = torch.randn((3))
            self.UNet.load_state_dict(unet_pth)
        
    def warp_color(self, IA_l, IB_lab, temperature=0.01):
        IA_rgb_from_gray = gray2rgb_batch(IA_l)
        with torch.no_grad():
            A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = self.vggnet(
                IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )
            B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = self.vggnet(
                IB_lab, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )

        # NOTE: output the feature before normalization
        # features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]
        # features_B = [B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1]
        # feature = {
        #     'A': features_A,
        #     'B': features_B
        # }
        
        A_relu2_1 = feature_normalize(A_relu2_1)
        A_relu3_1 = feature_normalize(A_relu3_1)
        A_relu4_1 = feature_normalize(A_relu4_1)
        A_relu5_1 = feature_normalize(A_relu5_1)
        B_relu2_1 = feature_normalize(B_relu2_1)
        B_relu3_1 = feature_normalize(B_relu3_1)
        B_relu4_1 = feature_normalize(B_relu4_1)
        B_relu5_1 = feature_normalize(B_relu5_1)

        nonlocal_BA_lab, similarity_map = self.WarpNet(
            IB_lab,
            A_relu2_1,
            A_relu3_1,
            A_relu4_1,
            A_relu5_1,
            B_relu2_1,
            B_relu3_1,
            B_relu4_1,
            B_relu5_1,
            temperature=temperature,
        )

        # return nonlocal_BA_lab, similarity_map, features_A # get coarse C
        return nonlocal_BA_lab # get coarse C(LAB)
    
    def forward(self, imgs_Input):
        # import ipdb; ipdb.set_trace()
        """
        Img_RGB: GT (0,1) 除了255
        Img_GREY: (0,255)
        Img_FAKE: (0,1) test中一张irrelavant的RGB
        """
        # import ipdb; ipdb.set_trace()
        Img_RGB = imgs_Input['img_RGB'].cuda() # torch.Size([1, 256, 256, 3])
        Img_GREY = color.rgb2gray(imgs_Input['img_RGB'])[:,:,:,np.newaxis] # torch.Size([1, 256, 256])
        Img_GREY = torch.from_numpy(np.concatenate((Img_GREY,Img_GREY,Img_GREY), axis=-1)).cuda()
        
        Img_FAKE_RGB = imgs_Input['img_FAKE'].cuda() # torch.Size([1, 256, 256, 3])
        
        Img_LAB = torch.from_numpy(color.rgb2lab(Img_RGB[0].cpu().detach().numpy())).cuda() # torch.Size([256, 256, 3])
        Img_L = Img_LAB[:, :, 0:1].permute(2,0,1) # torch.Size([1, 256, 256])
        Img_AB = Img_LAB[:, :, 1:3].permute(2,0,1) # torch.Size([2, 256, 256])
        
        Img_FAKE_LAB = torch.from_numpy(color.rgb2lab(Img_FAKE_RGB[0].cpu().detach().numpy())).cuda() # torch.Size([256, 256, 3])
        Img_FAKE_L = Img_FAKE_LAB[:, :, 0:1].permute(2,0,1)
        Img_FAKE_AB = Img_FAKE_LAB[:, :, 1:3].permute(2,0,1) # torch.Size([2, 256, 256])
        
        Mask = torch.randn(Img_L.shape).cuda() # torch.Size([1, 256, 256])
        thresh = 0.6
        Mask[Mask >= thresh] = 1.0
        Mask[Mask < thresh] = 0.0
        """
        distortion!
        腐蚀: 
        kernel = np.ones((3, 3), dtype=np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        """
        Img_Distortion_L = Img_L*(1-Mask) + Img_FAKE_L * Mask
        Img_Distortion_AB = Img_AB*(1-Mask) + Img_FAKE_AB*Mask # torch.Size([2, 256, 256])
        Img_Distortion_LAB = torch.cat([Img_Distortion_L, Img_Distortion_AB],0) # torch.Size([3, 256, 256])
        Img_Distortion_LAB_grey_AB = torch.cat([Img_L, Img_Distortion_AB],0) # torch.Size([3, 256, 256])
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # Img_Distortion_LAB = torch.from_numpy(cv2.erode(Img_Distortion_LAB.cpu().detach().numpy(), kernel, iterations=1)).cuda()
        # pred_lab = np.concatenate((Img_L.cpu().detach().numpy(), Img_Distortion_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0))
        
        pred_lab = np.concatenate((Img_Distortion_L.cpu().detach().numpy(), Img_Distortion_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0))
        Img_Ref_RGB = color.lab2rgb(pred_lab)
        # mg_Ref_RGB = lab2rgb_transpose(Img_L.cpu().detach().numpy(), Img_Distortion_AB.cpu().detach().numpy())
        # import ipdb; ipdb.set_trace()
        
        # Img_Coarse_LAB = self.warp_color(Img_L.unsqueeze(0), Img_Distortion_LAB.unsqueeze(0))
        Img_Coarse_LAB = Img_Distortion_LAB.unsqueeze(0)  # 简单版 LAB 都distortion
        # Img_Coarse_LAB = Img_Distortion_LAB_grey_AB.unsqueeze(0) # 简单版 只distort AB通道
        Img_Coarse_L = Img_Coarse_LAB[:, 0:1, :, :]
        Img_Coarse_AB = Img_Coarse_LAB[:, 1:3, :, :]
        Img_Coarse_L = Img_Coarse_L[0]
        Img_Coarse_AB = Img_Coarse_AB[0]
        Img_Coarse_L = torch.clamp(Img_Coarse_L, 0.0, 100.0) # stable? for lab2rgb_transpose
        Img_Coarse_AB = torch.clamp(Img_Coarse_AB, -100.0, 100.0)
        
        """
        用了clamp之后stable, no warning, 因为之后的 Img_Coarse_RGB = np.clip(color.lab2rgb(Img_Coarse_RGB), 0, 1)
        这步的数值上没有啥问题了, 如果直接用原图的L, 一开始会不稳定，但随着训练的进行, 会慢慢收敛
        """
        
        # 两张coarse组合方法
        Img_Coarse_RGB = np.concatenate((Img_L.cpu().detach().numpy(), Img_Coarse_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0)) # L
        # Img_Coarse_RGB = np.concatenate((Img_Coarse_L.cpu().detach().numpy(), Img_Coarse_AB.cpu().detach().numpy()), axis=0).transpose((1, 2, 0)) # distortion_L
        Img_Coarse_RGB = np.clip(color.lab2rgb(Img_Coarse_RGB), 0, 1)
        Img_Coarse_RGB = torch.from_numpy(Img_Coarse_RGB).float().cuda()
        TPS = self.batch_transform.exe(Img_Coarse_RGB.unsqueeze(0).permute(0,3,1,2))
        TPS_img = TPS['image'] # tps 变形太大
        TPS_future_img = TPS['future_image']
        TPS_mask = TPS['mask']
        Img_Coarse_RGB_TPS = Img_Coarse_RGB.unsqueeze(0).permute(0,3,1,2)*(1-TPS_mask) + TPS_future_img*TPS_mask
        mask = TPS_mask[0].permute(1,2,0).cpu().numpy()
        mask = np.concatenate((mask, mask, mask),axis=2)
        import ipdb; ipdb.set_trace()
        Img_Fine_RGB = self.UNet(Img_Coarse_RGB_TPS.cuda()).permute(0,2,3,1) # fine: (0,255) (batch_size(1), img_size, 3)

        model_output = {
            'Img_Fine_RGB': Img_Fine_RGB, # torch.Size([1, 256, 256, 3])  (0,1)
            'Img_RGB': Img_RGB, # torch.Size([1, 256, 256, 3]) (0,1)
            'Img_FAKE_RGB': Img_FAKE_RGB, # torch.Size([1, 256, 256, 3]) (0,1)
            'Img_L': Img_L, # (0,100)
            'Img_Ref': Img_Ref_RGB, # (0,1)
            'Img_Coarse_RGB': Img_Coarse_RGB_TPS, # (0,1)
            'TPS_mask': mask,
            'TPS_img': TPS_img,
            'TPS_future_img': TPS_future_img,
            'VGG19': self.vggnet
        }
        return model_output
        
        