# EVA6_Assignment_8

# Make a deep learning repo 

Working CIFAR DATA SET ; 

- Train 40 epoch 

- find 20 misclassified images 

- 20 GradCam output of misclassified images

- Apply augmentation of the images ( Random crop, cutout, rotate )

- Use Reduce LROn Plateau for learning rate 

- Layer normalization 

  #  

  #  Architecture  

  Layer (type)               Output Shape         Param # ================================================================            Conv2d-1           [-1, 64, 32, 32]           1,728         GroupNorm-2           [-1, 64, 32, 32]             128            Conv2d-3           [-1, 64, 32, 32]          36,864         GroupNorm-4           [-1, 64, 32, 32]             128            Conv2d-5           [-1, 64, 32, 32]          36,864         GroupNorm-6           [-1, 64, 32, 32]             128        BasicBlock-7           [-1, 64, 32, 32]               0            Conv2d-8           [-1, 64, 32, 32]          36,864         GroupNorm-9           [-1, 64, 32, 32]             128           Conv2d-10           [-1, 64, 32, 32]          36,864        GroupNorm-11           [-1, 64, 32, 32]             128       BasicBlock-12           [-1, 64, 32, 32]               0           Conv2d-13          [-1, 128, 16, 16]          73,728        GroupNorm-14          [-1, 128, 16, 16]             256           Conv2d-15          [-1, 128, 16, 16]         147,456        GroupNorm-16          [-1, 128, 16, 16]             256           Conv2d-17          [-1, 128, 16, 16]           8,192        GroupNorm-18          [-1, 128, 16, 16]             256       BasicBlock-19          [-1, 128, 16, 16]               0           Conv2d-20          [-1, 128, 16, 16]         147,456        GroupNorm-21          [-1, 128, 16, 16]             256           Conv2d-22          [-1, 128, 16, 16]         147,456        GroupNorm-23          [-1, 128, 16, 16]             256       BasicBlock-24          [-1, 128, 16, 16]               0           Conv2d-25            [-1, 256, 8, 8]         294,912        GroupNorm-26            [-1, 256, 8, 8]             512           Conv2d-27            [-1, 256, 8, 8]         589,824        GroupNorm-28            [-1, 256, 8, 8]             512           Conv2d-29            [-1, 256, 8, 8]          32,768        GroupNorm-30            [-1, 256, 8, 8]             512       BasicBlock-31            [-1, 256, 8, 8]               0           Conv2d-32            [-1, 256, 8, 8]         589,824        GroupNorm-33            [-1, 256, 8, 8]             512           Conv2d-34            [-1, 256, 8, 8]         589,824        GroupNorm-35            [-1, 256, 8, 8]             512       BasicBlock-36            [-1, 256, 8, 8]               0           Conv2d-37            [-1, 512, 8, 8]       1,179,648        GroupNorm-38            [-1, 512, 8, 8]           1,024           Conv2d-39            [-1, 512, 8, 8]       2,359,296        GroupNorm-40            [-1, 512, 8, 8]           1,024           Conv2d-41            [-1, 512, 8, 8]         131,072        GroupNorm-42            [-1, 512, 8, 8]           1,024       BasicBlock-43            [-1, 512, 8, 8]               0           Conv2d-44            [-1, 512, 8, 8]       2,359,296        GroupNorm-45            [-1, 512, 8, 8]           1,024           Conv2d-46            [-1, 512, 8, 8]       2,359,296        GroupNorm-47            [-1, 512, 8, 8]           1,024       BasicBlock-48            [-1, 512, 8, 8]               0           Linear-49                   [-1, 10]           5,130 ================================================================ Total params: 11,173,962 Trainable params: 11,173,962 Non-trainable params: 0 ---------------------------------------------------------------- Input size (MB): 0.01 Forward/backward pass size (MB): 13.50 Params size (MB): 42.63 Estimated Total Size (MB): 56.14

- Training accuracy = 85.185

- Test accuracy  = 86.59 
