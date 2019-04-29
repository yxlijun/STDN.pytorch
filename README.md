## STDN: Scale-Transferrable Object Detection ##
A PyTorch Implementation of Scale-Transferrable Object Detection，the official code is not found,so I trained the 
model with pytorch,the code support: 

  * Support for the MS COCO dataset and VOC PASCAL dataset
  * Support for stdn300,stdn321,stdn513 training and testing
  * Support for mulltigpu training
  * Support training and and testing in VOC and COCO 
 
>> because my GPU is limited,so I only train the stdn300 in VOC0712,if your gpu is enough,you can train other model according to configs/*,the model can be downloaded in [stdn300_densenet169](https://drive.google.com/file/d/1msbqNYSTppVCCsAIHfuA-70dzopIITNQ/view?usp=sharing)。the map is 76.30,the map is lower because I have not  pretrained model.

### MAP in VOC2007

| 	Original|   Ours	|
| --------- |-----------|
|	78.1    |    76.30  |


### Preparation
**the supported version is pytorch-0.4.1 or pytorch-1.0**  
* tqdm
* opencv
* addict
* pytorch>=0.4

- Clone this repository.
```Shell
git clone https://github.com/qijiezhao/M2Det.git
```
- Compile the nms and coco tools:

```Shell
sh make.sh
```

- Prepare dataset (e.g., VOC, COCO), refer to [sss.pytorch](https://github.com/amdegroot/ssd.pytorch) for detailed instructions.

### train
you can train different set according to configs/*  
```
	python train.py --dataset VOC\COCO --config ./configs/stdn300_densenet169.py  
```  
if you train with multi gpu    
```  
  CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset VOC\COCO --config ./configs/stdn300_densenet169.py --ngpu 2
```

### eval
you can evaluate your model in  voc and coco  
```
	python test.py --dataset VOC\COCO --trained_model ./weights/STDN_VOC_size300_netdensenet_epoch650.pth 
```

### demo 
you can test your image  
```
 python demo.py --dataset VOC\COCO --trained_model ./weights/STDN_VOC_size300_netdensenet_epoch650.pth --show  
```




