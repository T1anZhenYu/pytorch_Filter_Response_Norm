# pytorch_Filter_Response_Norm
### usage
FRN:

`python cifar.py --arch resnet_frn --depth 20`

BN:

`python cifar.py --arch resnet --depth 20`


reproduce FRN on resnet_20/cifar_10

### result
| images per gpu|128 |32|
| ------ | ------ |------ | 
| batch_norm | 93.01% |91.3%|
| FRN | 92.3% | 92.3%|

