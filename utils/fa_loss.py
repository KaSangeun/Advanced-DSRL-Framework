import torch

class FALoss(torch.nn.Module):
    def __init__(self, subscale=0.125): # 0.0625
        super(FALoss, self).__init__()
        self.subscale = int(1 / subscale) # 16 (In the paper, subscale value is 8)

    def forward(self, feature1, feature2):
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1) # fea_Seg [64 x 128 x 3]
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2) # fea_sr [64 x 128 x 3]
        
        # similarity matrix of fea_seg 
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, C, width * height)  # [N, C, W*H]
        L2norm1 = torch.norm(feature1, 2, 1, keepdim=True)  # [N, 1, W*H]
        L2norm1 = L2norm1.repeat(1, C, 1)  # [N, C, W*H]
        feature1 = feature1 / (L2norm1 + 1e-8)  # Avoid division by zero
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)  # [N, W*H, W*H]

        # similarity matrix of fea_sr
        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, C, width * height)  # [N, C, W*H]
        L2norm2 = torch.norm(feature2, 2, 1, keepdim=True)  # [N, 1, W*H]
        L2norm2 = L2norm2.repeat(1, C, 1)  # [N, C, W*H]
        feature2 = feature2 / (L2norm2 + 1e-8)  # Avoid division by zero
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)  # [N, W*H, W*H]

        L1norm = torch.norm(mat2 - mat1, 1)

        return L1norm / ((height * width) ** 2)
