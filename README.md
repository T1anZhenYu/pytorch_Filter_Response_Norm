# pytorch_Filter_Response_Norm
### usage
BN:(for one GPU)

`python cifar.py --arch resnet --depth 20 --epoch 400 --train-batch 32`

FRN:

`python cifar.py --arch resnet_frn --depth 20 --epoch 400 --cos true --train-batch 32 --ramp-up 50`

Max_Min_FRN:

`python cifar.py --arch resnet_max_min_frn --depth 20 --epoch 400 --cos true --train-batch 32 --ramp-up 50`




reproduce FRN on resnet_20/cifar_10

### result
In paper, 32 images per gpu is the biggest batch size. So the reproduction matches paper's conclusion.

| images per gpu|128 |32|
| ------ | ------ |------ | 
| batch_norm | 93.01% |91.2%|
| FRN | 92.3% | 92.3%|

### Experments report
`https://blog.csdn.net/qq_18053809/article/details/103478732`

