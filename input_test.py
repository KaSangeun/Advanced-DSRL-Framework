import os
import numpy as np
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)  # input image
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)  # label of image 

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_width=self.args.crop_width, crop_height=self.args.crop_height, fill=255), # annotation
            tr.RandomGaussianBlur(),
            #tr.FixedResize(size=(512, 256)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixScaleCrop(crop_width=self.args.crop_width, crop_height=self.args.crop_height),
            #tr.FixedResize(size=(512, 256)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixedResize(size=self.args.crop_size),
            tr.FixedResize(size=(1024, 512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 1025 # 513
   #args.crop_size = 513

    #cityscapes_train = CityscapesSegmentation(args, split='train')

    #dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    #fixed_resize = tr.FixedResize(size=(512, 512))

    img_path = 'newdata/why/cityscapes/gtFine/train/erfurt/erfurt_000000_000019_gtFine_color.png'
    lbl_path = 'newdata/why/cityscapes/gtFine/train/erfurt/erfurt_000000_000019_gtFine_labelIds.png'

    _img = Image.open(img_path).convert('RGB')
    _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
    _tmp = CityscapesSegmentation(args).encode_segmap(_tmp)
    _target = Image.fromarray(_tmp)

    sample = {'image': _img, 'label': _target}

    print('Original image size: ', _img.size)  # _img.size는 (width, height) 형태

    #fixed_resize2 = tr.FixScaleCrop(crop_width=1024, crop_height=512)
    #fixed_resize2 = tr.FixedResize(size=(1024,512))
    fixed_resize2 = tr.RandomScaleCrop(base_size=1024, crop_width=1024, crop_height=512, fill=255)
    sample_resized = fixed_resize2(sample)

    resized_img = sample_resized['image']
    print(f'Resized image size: {resized_img.size}')

    img = np.array(sample_resized['image'])
    gt = np.array(sample_resized['label'])
    tmp = np.array(gt).astype(np.uint8)
    segmap = decode_segmap(tmp, dataset='cityscapes')
    img_tmp = img.astype(np.uint8)

    plt.figure()
    plt.title('display')
    plt.subplot(211)
    plt.imshow(img_tmp)
    plt.subplot(212)
    plt.imshow(segmap)
    plt.show(block=True)

    # if you wanna select img randomly, remove those two annotations and comment out upper codes

    '''
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'][jj]
            gt = sample['label'][jj]

            # Convert tensor to PIL Image for the fixed resize transform
            img = transforms.ToPILImage()(img)
            gt = transforms.ToPILImage()(gt)

            # Create a sample dictionary
            sample_resized = {'image': img, 'label': gt}

            # Apply fixed resize
            sample_resized = fixed_resize(sample_resized)

            resized_img = sample_resized['image']
            print(f'Resized image size: {resized_img.size}') 

            # Convert back to numpy for visualization
            img = np.array(sample_resized['image'])
            gt = np.array(sample_resized['label'])
            tmp = np.array(gt).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = img.astype(np.uint8)

            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
    '''
