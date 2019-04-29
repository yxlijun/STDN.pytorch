## STDN: Scale-Transferrable Object Detection ##
A PyTorch Implementation of Scale-Transferrable Object Detection，the official code is not found,so I trained the 
model with pytorch,the code support: 

  * Support for the MS COCO dataset and VOC PASCAL dataset
  * Support for stdn300,stdn321,stdn513 training and testing
  * Support for mulltigpu training
  * Support training and and testing in VOC and COCO 
 
>> because my GPU is limited,so I only train the stdn300 in VOC0712,if your gpu is enough,you can train other model according to configs/*,the model can be downloaded in [stdn300_densenet169](https://drive.google.com/file/d/1msbqNYSTppVCCsAIHfuA-70dzopIITNQ/view?usp=sharing)。the map is 76.30,the map is lower because I have not  pretrained model.

### MAP in VOC2007
| Original | Ours | 
|:-:|:-:|:-:|:-:| 
| 78.1 % | 76.30 |

