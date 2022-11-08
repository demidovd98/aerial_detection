# Aerial_object_detection

Trained model weights - https://drive.google.com/file/d/1QfDVX-l3TrSN23-v0Kj_kX2gq4HhEG_K/view?usp=sharing

Used detectron2-0.6

# Installation instructions

Folllowed same instructions as detectron2

Please use the detectron2_env.yml file to use the same envrionment used for experiments

Experiments have been run with cuda-11.1 on Quadro RTX-6000

# For training

Run train.py file

# For inference

Run inference.py file

# Modifications performed

1. Changed detectron2/detectron2/modeling/backbone/fpn.py file to include channel attention and weigh output feature maps
2. Changed detectron2/detectron2/modeling/sampling.py to include class balanced sampler
3. Changed detectron2/detectron2/modeling/proposal_generator/rpn.py to include density prediction head and smooth l1 loss
