import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.fa_loss import FALoss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import matplotlib.pyplot as plt
import torchvision.utils as vutils

torch.cuda.empty_cache()

w_sr=0.1 # 0.1
w_fa=1.0 # 1.0

'''
def compare_weights(state_dict1, state_dict2):
    match = True
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f'Weight mismatch found at layer: {key}')
            match = False
    if match:
        print('Weights match!')
    return match

def visualize_weights(epoch, model, layer_name, nrow=8, padding=1, file_path=None):
    
    # check module name
    #for name, module in model.named_modules():
        #print(name)
     # Find the layer by name
    layer = dict(model.named_modules())[layer_name]
    
    # Extract the weights
    weights = layer.weight.data.cpu()
    
    # Normalize the weights
    min_w = torch.min(weights)

    normalized_weights = (-1 / (2 * min_w)) * weights + 0.5
    # Add an extra dimension using unsqueeze
    grid = vutils.make_grid(normalized_weights.unsqueeze(1), nrow=nrow, padding=padding, normalize=True, scale_each=True)

    #normalized_weights = (-1 / (2 * min_w)) * weights + 0.5
    
    # Create a grid of weights
    #grid = vutils.make_grid(normalized_weights.unsqueeze(1), nrow=nrow, padding=padding, normalize=True, scale_each=True)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f'Epoch {epoch} - {layer_name} Weights')
    plt.axis('off')
    
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
'''

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type) # Cross Entropy Loss
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        # added code 5  
        # Save initial weights
        #initial_weights = self.model.state_dict()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            # if not args.ft:
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            
            # added code 3
            # Save loaded weights
            #loaded_weights = self.model.state_dict()

            # Compare weights
            #compare_weights(initial_weights, loaded_weights)
        #else:
            #loaded_weights = None

        '''file_path = 'loaded_init_weights.pth'
        torch.save(loaded_weights, file_path)
        print(f"loaded_weights saved to {file_path}")'''

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
    '''
    # added code 1
    # Save model weights after loading checkpoint
    def save_weight_summary_to_file(self, model, file_path):
        with open(file_path, 'w') as f:
            for name, param in model.named_parameters():
                mean = param.data.mean().item()
                std = param.data.std().item()
                f.write(f"{name}: mean={mean}, std={std}\n")
    '''

    '''def save_model_weights(self, epoch, prefix):
        model_dir = os.path.join(self.saver.experiment_dir, 'weights')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f'{prefix}_model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), model_path)'''
    

    def cross_attention(self, query, key, value):
        # query, key, value shapes: [batch_size, channels, height, width]
        batch_size, channels_q, height, width = query.size()
        channels_kv = key.size(1)

        # Reshape to [batch_size, channels, height*width] for attention calculation
        query = query.view(batch_size, channels_q, -1)  # [batch_size, channels_q, H*W]
        key = key.view(batch_size, channels_kv, -1)     # [batch_size, channels_kv, H*W]
        value = value.view(batch_size, channels_kv, -1) # [batch_size, channels_kv, H*W]

        # Transpose key for dot product: [batch_size, H*W, channels_kv]
        key = key.transpose(1, 2)

        # Calculate attention scores: [batch_size, channels_q, channels_kv]
        attention_scores = torch.bmm(query, key)  # [batch_size, channels_q, H*W]

        # Normalize scores using softmax
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention to the value: [batch_size, channels_q, H*W]
        attention_output = torch.bmm(attention_scores, value)

        # Reshape back to original feature map shape: [batch_size, channels_q, height, width]
        attention_output = attention_output.view(batch_size, channels_q, height, width)

        return attention_output
    

    def training(self, epoch):
        # added code 1 - for checking weights
        ##self.save_model_weights(epoch, prefix='before_training')

        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader) 
        num_img_tr = len(self.train_loader) # batch size

        # added code 2
        # Save model weights before training
        #self.save_weight_summary_to_file(self.model, f'weight_summary_before_epoch_{epoch}.txt')

        '''loaded_weights = self.model.state_dict()
        file_path = 'loaded_init_weights.pth'
        torch.save(loaded_weights, file_path)
        print(f"loaded_weights saved to {file_path}")'''


        for i, sample in enumerate(tbar): # training 1 Epoch, i: batch index
            image, target = sample['image'], sample['label'] # image.shape: torch.Size([3, 3, 256, 512]) -> [batch_size, channels, height, width]
            #print("Shape of the tensor1:", image.shape) # [3, 3, 256,512]
            #print("Shape of the tensor2:", target.shape) # [3, 256, 512]
            input_img=torch.nn.functional.interpolate(image,size=[i//2 for i in image.size()[2:]], mode='bilinear', align_corners=True) # image width, height //2 (changed bilinear to bicubic)
            if self.args.cuda:
                input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            #print("input image size: ", input_img.shape) # torch.Size([2, 3, 512, 512]) => smaller than the original image 
            output,output_sr,fea_seg,fea_sr = self.model(input_img) # [1024 x 2048 x num_classes], [1024 x 2048 x 3], [1024 x 2048 x 3]. [1024 x 2048 x 3]
            """print("output shape: ", output.size()) # [512 x 1024 x num_classes]
            print("output_sr shape", output_sr.size()) # [512 x 1024 x 3]
            print("target shape", target.size()) # [512 x 1024]
            print("fea_seg shape", fea_seg.size()) # [256 x 512 x num_classes]
            print("fea_sr shape", fea_sr.size()) # [256 x 512 x 32]"""

            """ Apply Cross-Attention """
            cross_attention_map = self.cross_attention(fea_seg, fea_sr, fea_sr) # querry, key, value
            fused_fea = fea_seg + cross_attention_map # feature1 + cross_attention_map
            #pred_ca = F.interpolate(fused_fea,size=[2*i for i in input.size()[2:]], mode='bicubic', align_corners=True) # adjust shape of fused_fea to output of SSSR 
            pred_ca = pred_ca = F.interpolate(fused_fea, size=(512, 1024), mode='bilinear', align_corners=True)

            # code modify
            criterion_seg = self.criterion(output, target)
            criterion_sr = torch.nn.MSELoss()(output_sr, image)
            criterion_ca = self.criterion(pred_ca, target)
            #criterion_fa = FALoss()(fea_seg, fea_sr)
            #loss = criterion_seg + w_sr * criterion_sr + w_fa * criterion_fa
            loss = criterion_seg + w_sr * criterion_sr + w_fa * criterion_ca

            #print("Shape of the tensor2:", image.shape)
            #print("Shape of the output sr:", output_sr.shape)
            #print("Shape of the feature sr:", fea_sr.shape)

            #print('SS Loss: %3f, SR Loss: %3f, FA Loss: %3f' % (criterion_seg.item(), criterion_sr.item(), criterion_fa.item()))


            #print(f"Segmentation Loss: {criterion_seg.item()}")
            #print(f"SR Loss: {criterion_sr.item()}")
            #print(f"FA Loss: {criterion_fa.item()}")

            #loss = self.criterion(output, target)+w_sr*torch.nn.MSELoss()(output_sr,image)+w_fa*FALoss()(fea_seg,fea_sr)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            #print('Epoch: %d, batch: %d/%d' % (epoch, i, self.args.batch_size))
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        # added code 3
        # Save model weights after training
        ##self.save_weight_summary_to_file(self.model, f'weight_summary_after_epoch_{epoch}.txt')


        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('batch size: ', self.args.batch_size)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss) # sum of train_loss

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


        # added code 2 - for checking weights
        #self.save_model_weights(epoch, prefix='after_training')
        '''visualize_weights(epoch, self.model, 'module.backbone.conv1', file_path='module.backbone.conv1.png')
        visualize_weights(epoch, self.model, 'module.backbone.layer4.2.conv1', file_path='backbone.layer4.2.conv3.png')
        visualize_weights(epoch, self.model, 'aspp.aspp1.atrous_conv', file_path='aspp.aspp1.atrous_conv.png')
        visualize_weights(epoch, self.model, 'aspp.conv1', file_path='aspp.conv1.png')
        visualize_weights(epoch, self.model, 'decoder.conv1', file_path='decoder.conv1.png')
        visualize_weights(epoch, self.model, 'module.decoder.last_conv.8', file_path='module.decoder.last_conv.8.png')
        visualize_weights(epoch, self.model, 'module.sr_decoder.conv1', file_path='module.sr_decoder.conv1.png')
        visualize_weights(epoch, self.model, 'module.sr_decoder.last_conv.8', file_path='module.sr_decoder.last_conv.8.png')
        visualize_weights(epoch, self.model, 'module.pointwise.1', file_path='module.pointwise.1.png')
        visualize_weights(epoch, self.model, 'module.up_sr_1', file_path='module.up_sr_1.png')
        visualize_weights(epoch, self.model, 'module.up_edsr_1.conv.0', file_path='module.up_edsr_1.conv.0.png')
        visualize_weights(epoch, self.model, 'module.up_edsr_3.residual_upsampler.0', file_path='module.up_edsr_3.residual_upsampler.0.png')
        visualize_weights(epoch, self.model, 'module.up_sr_3', file_path='module.up_sr_3.png')
        visualize_weights(epoch, self.model, 'module.up_conv_last', file_path='module.up_conv_last.png')'''


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r') # changed val_loader to test_loader
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            input_img=torch.nn.functional.interpolate(image,size=[i//2 for i in image.size()[2:]], mode='bicubic', align_corners=True) # changed bilinear to bicubic
            if self.args.cuda:
                input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()
            with torch.no_grad():
                output,_,_,_ = self.model(input_img)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--crop-width', type=int, default=1024,
                        help='crop image width size')
    parser.add_argument('--crop-height', type=int, default=512,
                        help='crop image height size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200, # 1000 -> 400 -> 100 -> 10
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids) 

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.001, # 0.005 -> 0.001 
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    #print('Starting Epoch:', trainer.args.start_epoch)
    #print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
