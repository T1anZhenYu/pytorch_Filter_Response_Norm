# pytorch_Filter_Response_Norm
### usage
FRN:

`python cifar.py --arch resnet_frn --depth 20`

BN:

`python cifar.py --arch resnet --depth 20`


reproduce FRN on resnet_20/cifar_10

### result

| Layer type |accuracy |
| ------ | ------ | 
| batch_norm | 93.01% | 
| FRN | 92.5% | 
