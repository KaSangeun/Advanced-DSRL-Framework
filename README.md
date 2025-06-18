# Advanced DSRL(Dual Super-Resolution Learning) Framework
Enhanced semantic segmentation performance by introducing the Advanced DSRL framework using Attention Mechanism. <br/>

## Duration
2024.06.24 ~ 2024.08.16

## Main Idea
Cross Attention Score 계산을 통해 SISR(Single Image Super-Resolution) feature에서 의미 있는 정보를 가져와 SSSR(Semantic Segmentation Super-Resolution) feature를 보완함으로써 semantic segmentation 성능을 향상시킴 <br/>
By computing Cross Attention Scores, meaningful information from SISR features is extracted to enhance SSSR features, thereby improving semantic segmentation performance. <br/><br/>
→ 기존 DSRL에서의 FA Loss를 보완된 SSSR feature의 segmentation 성능으로 대체함
The original FA Loss in DSRL is replaced with the segmentation performance of the enhanced SSSR features.

더 자세한 사항은 [UROP 소논문](https://github.com/KaSangeun/Advanced-DSRL-Framework/blob/main/UROP_short_paper.pdf)을 통해 확인할 수 있다. <br/>
For more details, please refer to the [short paper from UROP](https://github.com/KaSangeun/Advanced-DSRL-Framework/blob/main/UROP_short_paper.pdf).

## Run
1. Modify your dataset path in [`mypath.py`](https://github.com/KaSangeun/Advanced-DSRL-Framework/blob/main/mypath.py)

2. Run the training script:

   - To train from scratch:
     ```shell
     python train.py
     ``` 
   - To resume training from pre-trained weights:  
     ```shell
     python train.py --resume '[path to your checkpoint file]'
     ```
     You can download pre-trained weights from [here](https://drive.google.com/file/d/1qkThLgs3vPGj8Dc7E6Bpt983YpbrCeFx/view?usp=sharing).

## Based on
This project is based on [Dootmaan-DSRL](https://github.com/Dootmaan/DSRL), which is licensed under the [MIT License](https://github.com/KaSangeun/Advanced-DSRL-Framework/blob/main/LICENSE).
