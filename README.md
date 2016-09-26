# Stacked Hourglass Networks for Human Pose Estimation (Training Code)

This is the training pipeline used for:

Alejandro Newell, Kaiyu Yang, and Jia Deng,
**Stacked Hourglass Networks for Human Pose Estimation**,
[arXiv:1603.06937](http://arxiv.org/abs/1603.06937), 2016.

A pretrained model is available on the [project site](http://www-personal.umich.edu/~alnewell/pose). You can use the option `-loadModel path/to/model` to try fine-tuning. 

To run this code, make sure the following are installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5
- cudnn

## Getting Started ##

Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de), and place the `images` directory in `data/mpii`. From there, it is as simple as running `th main.lua -expID test-run` (the experiment ID is arbitrary). To run on [FLIC](http://bensapp.github.io/flic-dataset.html), again place the images in a directory `data/flic/images` then call `th main.lua -dataset flic -expID test-run`.

Most of the command line options are pretty self-explanatory, and can be found in `src/opts.lua`. The `-expID` option will be used to save important information in a directory like `pose-hg-train/exp/mpii/test-run`. This directory will include snapshots of the trained model, training/validations logs with loss and accuracy information, and details of the options set for that particular experiment.

## Running experiments ##

There are a couple features to make experiments a bit easier:

- Experiment can be continued with `th main.lua -expID example-exp -continue` it will pick up where the experiment left off with all of the same options set. But let's say you want to change an option like the learning rate, then you can do the same call as above but add the option `-LR 1e-5` for example and it will preserve all old options except for the new learning rate.

- In addition, the `-branch` option allows for the initialization of a new experiment directory leaving the original experiment intact. For example, if you have trained for a while and want to drop the learning rate but don't know what to change it to, you can do something like the following: `th main.lua -branch old-exp -expID new-exp-01 -LR 1e-5` and then compare to a separate experiment `th main.lua -branch old-exp -expID new-exp-02 -LR 5e-5`.

In `src/misc` there's a simple script for monitoring a set of experiments to visualize and compare training curves.

#### Getting final predictions ####

To generate final test set predictions for MPII, you can call:

`th main.lua -branch your-exp -expID final-preds -finalPredictions -nEpochs 0`

This assumes there is an experiment that has already been run. If you just want to provide a pre-trained model, that's fine too and you can call:

`th main.lua -expID final-preds -finalPredictions -nEpochs 0 -loadModel /path/to/model`

#### Training accuracy metric ####

For convenience during training, the accuracy function evaluates PCK by comparing the output heatmap of the network to the ground truth heatmap. The normalization in this case will be slightly different than the normalization done when officially evaluating on FLIC or MPII. So there will be some discrepancy between the numbers, but the heatmap-based accuracy still provides a good picture of how well the network is learning during training.

## Final notes ##

In the paper, the training time reported was with an older version of cuDNN, and after switching to cuDNN 4, training time was cut in half. Now, with a Titan X NVIDIA GPU, training time from scratch is under 3 days for MPII, and about 1 day for FLIC.

#### pypose/ ####

Included in this repository is a folder with a bunch of old python code that I used. It hasn't been updated in a while, and might not be totally functional at the moment. There are a number of useful functions for doing evaluation and analysis on pose predictions and it is worth digging into. It will be updated and cleaned up soon.

#### Questions? ####

I am sure there is a lot not covered in the README at the moment so please get in touch if you run into any issues or have any questions!

## Acknowledgements ##

Thanks to Soumith Chintala, this pipeline is largely built on his example ImageNet training code available at:
[https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
