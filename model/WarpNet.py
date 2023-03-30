import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class WTA_scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out

class WarpNet(nn.Module):
    """ input is Al, Bl, channel = 1, range~[0,255] """

    def __init__(self, batch_size):
        super(WarpNet, self).__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
        )

        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        B_lab_map,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=0.001 * 5,
        detach_flag=False,
        WTA_scale_weight=1,
        feature_noise=0,
    ):
        batch_size = B_lab_map.shape[0]
        channel = B_lab_map.shape[1]
        image_height = B_lab_map.shape[2]
        image_width = B_lab_map.shape[3]
        feature_height = int(image_height / 4)
        feature_width = int(image_width / 4)

        # scale feature size to 44*44
        A_feature2_1 = self.layer2_1(A_relu2_1)
        B_feature2_1 = self.layer2_1(B_relu2_1)
        A_feature3_1 = self.layer3_1(A_relu3_1)
        B_feature3_1 = self.layer3_1(B_relu3_1)
        A_feature4_1 = self.layer4_1(A_relu4_1)
        B_feature4_1 = self.layer4_1(B_relu4_1)
        A_feature5_1 = self.layer5_1(A_relu5_1)
        B_feature5_1 = self.layer5_1(B_relu5_1)

        # concatenate features
        if A_feature5_1.shape[2] != A_feature2_1.shape[2] or A_feature5_1.shape[3] != A_feature2_1.shape[3]:
            A_feature5_1 = F.pad(A_feature5_1, (0, 0, 1, 1), "replicate")
            B_feature5_1 = F.pad(B_feature5_1, (0, 0, 1, 1), "replicate")
        # import ipdb; ipdb.set_trace()
        """
        bug?
        """
        try:
            A_features = self.layer(torch.cat((A_feature2_1, A_feature3_1, A_feature4_1, A_feature5_1), 1))
            B_features = self.layer(torch.cat((B_feature2_1, B_feature3_1, B_feature4_1, B_feature5_1), 1))
        except:
            import ipdb; ipdb.set_trace()
        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(B_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

        # downsample the reference color
        B_lab = F.avg_pool2d(B_lab_map, 4)
        B_lab = B_lab.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)  # 2*1936*channel

        # multiply the corr map with color
        y = torch.matmul(f_div_C, B_lab)  # 2*1936*channel
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        y = self.upsampling(y)
        similarity_map = self.upsampling(similarity_map)

        return y, similarity_map