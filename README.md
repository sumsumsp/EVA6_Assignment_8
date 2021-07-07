# EVA6_Assignment_8

# Make a deep learning repo 

Working CIFAR DATA SET  and resnet18 (); 

- Train 40 epoch 

- find 20 misclassified images 

- 20 GradCam output of misclassified images

- Apply augmentation of the images ( Random crop, cutout, rotate )

- Use Reduce LROn Plateau for learning rate 

- Layer normalization 

  #  

  #  Model 
    Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
        GroupNorm-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
        GroupNorm-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
        GroupNorm-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
        GroupNorm-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
        GroupNorm-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
        GroupNorm-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
        GroupNorm-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
        GroupNorm-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
        GroupNorm-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
        GroupNorm-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.50
Params size (MB): 42.63
Estimated Total Size (MB): 56.14

# Training Log 
 0%|          | 0/391 [00:00<?, ?it/s]Epoch 0
loss=1.8676389455795288 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0158, Accuracy: 12127/50000 (24.25%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -1.0586, Accuracy: 3165/10000 (31.65%)

Epoch 1
loss=1.6754270792007446 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.85it/s]
Train set: Average loss: 0.0137, Accuracy: 17560/50000 (35.12%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -1.7338, Accuracy: 3848/10000 (38.48%)

Epoch 2
loss=1.8621248006820679 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.85it/s]
Train set: Average loss: 0.0126, Accuracy: 20467/50000 (40.93%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -2.0218, Accuracy: 4494/10000 (44.94%)

Epoch 3
loss=1.4589837789535522 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.85it/s]
Train set: Average loss: 0.0117, Accuracy: 22646/50000 (45.29%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -2.5638, Accuracy: 5198/10000 (51.98%)

Epoch 4
loss=1.3053028583526611 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.85it/s]
Train set: Average loss: 0.0109, Accuracy: 24521/50000 (49.04%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -2.8499, Accuracy: 5550/10000 (55.50%)

Epoch 5
loss=1.4055052995681763 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0102, Accuracy: 26328/50000 (52.66%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -3.1096, Accuracy: 5772/10000 (57.72%)

Epoch 6
loss=1.4074070453643799 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0096, Accuracy: 27838/50000 (55.68%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -3.5597, Accuracy: 5810/10000 (58.10%)

Epoch 7
loss=1.04705011844635 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0092, Accuracy: 28900/50000 (57.80%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -3.5050, Accuracy: 6333/10000 (63.33%)

Epoch 8
loss=0.9893921613693237 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0084, Accuracy: 30565/50000 (61.13%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.0099, Accuracy: 6439/10000 (64.39%)

Epoch 9
loss=0.8171995878219604 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0082, Accuracy: 31363/50000 (62.73%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.0754, Accuracy: 6343/10000 (63.43%)

Epoch 10
loss=0.9380043745040894 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0076, Accuracy: 32542/50000 (65.08%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.2132, Accuracy: 6933/10000 (69.33%)

Epoch 11
loss=0.790594756603241 batch_id=390: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s]
Train set: Average loss: 0.0073, Accuracy: 33478/50000 (66.96%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.6615, Accuracy: 7034/10000 (70.34%)

Epoch 12
loss=0.7585653066635132 batch_id=390: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s]
Train set: Average loss: 0.0069, Accuracy: 34569/50000 (69.14%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.9965, Accuracy: 7108/10000 (71.08%)

Epoch 13
loss=0.7387370467185974 batch_id=390: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s]
Train set: Average loss: 0.0066, Accuracy: 35207/50000 (70.41%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.9983, Accuracy: 7303/10000 (73.03%)

Epoch 14
loss=0.7266136407852173 batch_id=390: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s]
Train set: Average loss: 0.0064, Accuracy: 35566/50000 (71.13%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.1594, Accuracy: 7487/10000 (74.87%)

Epoch 15
loss=0.9985149502754211 batch_id=390: 100%|██████████| 391/391 [02:19<00:00,  2.80it/s]
Train set: Average loss: 0.0061, Accuracy: 36220/50000 (72.44%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.2249, Accuracy: 7477/10000 (74.77%)

Epoch 16
loss=0.7925717830657959 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0059, Accuracy: 36891/50000 (73.78%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.2160, Accuracy: 7456/10000 (74.56%)

Epoch 17
loss=0.5734909772872925 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0057, Accuracy: 37308/50000 (74.62%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.5439, Accuracy: 7621/10000 (76.21%)

Epoch 18
loss=0.9711763262748718 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0056, Accuracy: 37377/50000 (74.75%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.5501, Accuracy: 7767/10000 (77.67%)

Epoch 19
loss=0.6186760663986206 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0054, Accuracy: 37964/50000 (75.93%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.7334, Accuracy: 7705/10000 (77.05%)

Epoch 20
loss=0.7190467119216919 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0053, Accuracy: 38091/50000 (76.18%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.7141, Accuracy: 7686/10000 (76.86%)

Epoch 21
loss=0.6499706506729126 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0051, Accuracy: 38618/50000 (77.24%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.8877, Accuracy: 8103/10000 (81.03%)

Epoch 22
loss=0.7031919956207275 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0050, Accuracy: 38853/50000 (77.71%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.9398, Accuracy: 7755/10000 (77.55%)

Epoch 23
loss=0.6404942870140076 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s]
Train set: Average loss: 0.0048, Accuracy: 39233/50000 (78.47%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.9085, Accuracy: 7945/10000 (79.45%)

Epoch 24
loss=0.5424283146858215 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0047, Accuracy: 39525/50000 (79.05%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.3239, Accuracy: 8240/10000 (82.40%)

Epoch 25
loss=0.5381278395652771 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0046, Accuracy: 39729/50000 (79.46%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.5818, Accuracy: 8257/10000 (82.57%)

Epoch 26
loss=0.5153371095657349 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0044, Accuracy: 40046/50000 (80.09%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.4508, Accuracy: 8221/10000 (82.21%)

Epoch 27
loss=0.7344198822975159 batch_id=390: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
Train set: Average loss: 0.0043, Accuracy: 40280/50000 (80.56%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.6728, Accuracy: 8323/10000 (83.23%)

Epoch 28
loss=0.5397557020187378 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0042, Accuracy: 40473/50000 (80.95%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.6744, Accuracy: 8068/10000 (80.68%)

Epoch 29
loss=0.5390773415565491 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0042, Accuracy: 40625/50000 (81.25%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.7473, Accuracy: 8052/10000 (80.52%)

Epoch 30
loss=0.6645799279212952 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0040, Accuracy: 41002/50000 (82.00%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.7636, Accuracy: 8186/10000 (81.86%)

Epoch 31
loss=0.5915881395339966 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0041, Accuracy: 40843/50000 (81.69%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9258, Accuracy: 8230/10000 (82.30%)

Epoch 32
loss=0.597694993019104 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0039, Accuracy: 41341/50000 (82.68%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9618, Accuracy: 8455/10000 (84.55%)

Epoch 33
loss=0.32283303141593933 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0038, Accuracy: 41550/50000 (83.10%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.1807, Accuracy: 8340/10000 (83.40%)

Epoch 34
loss=0.5911712050437927 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0038, Accuracy: 41643/50000 (83.29%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.2875, Accuracy: 8423/10000 (84.23%)

Epoch 35
loss=0.35656243562698364 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0037, Accuracy: 41770/50000 (83.54%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.2289, Accuracy: 8447/10000 (84.47%)

Epoch 36
loss=0.47242388129234314 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0035, Accuracy: 42064/50000 (84.13%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.4035, Accuracy: 8468/10000 (84.68%)

Epoch 37
loss=0.4657928943634033 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0035, Accuracy: 42202/50000 (84.40%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.3786, Accuracy: 8341/10000 (83.41%)

Epoch 38
loss=0.433002233505249 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
Train set: Average loss: 0.0035, Accuracy: 42248/50000 (84.50%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.2720, Accuracy: 8403/10000 (84.03%)

Epoch 39
loss=0.44349947571754456 batch_id=390: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s]
Train set: Average loss: 0.0033, Accuracy: 42591/50000 (85.18%)



Test set: Average loss: -7.7030, Accuracy: 8659/10000 (86.59%)

- Training accuracy = 85.185

- Test accuracy  = 86.59 
