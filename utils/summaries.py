import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

from torchvision.utils import save_image
from datetime import datetime


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
        

        '''# Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save Image
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        img_name = f'{timestamp}_image.png'
        save_image(grid_image, img_name)  # 이미지 저장
        writer.add_image('Image', grid_image, global_step)

        # Save Predicted Label
        pred_label = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        pred_name = f'{timestamp}_predicted_label.png'
        save_image(grid_image, pred_name)  # 예측된 라벨 저장
        writer.add_image('Predicted label', grid_image, global_step)


        # Save Groundtruth Label
        true_label = decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        true_name = f'{timestamp}_groundtruth_label.png'
        save_image(grid_image, true_name)  # 실제 라벨 저장
        writer.add_image('Groundtruth label', grid_image, global_step)'''