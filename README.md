# Implementation of an U-NET

Implemetation of the U-NET Structure. Built based on https://www.youtube.com/watch?v=IHq1t7NxS8k and modified further.

## Usage
The image shape must be `(n, m, ny, nx)` where
- `n`: number of images
- `m`: image depts (e.g. rgb=3, gray=1)
- `ny`: vertical image size [pixel]
- `nx`: horizontal image size [pixel]

The label shape must be `(n, m, ny, nx)` where
- `n`: number of images
- `m`: number of features, typically 1
- `ny`: vertical image size [pixel]
- `nx`: horizontal image size [pixel]

Besides the obvious parameters like `device`, `batch_size` etc, 
you can **adjust/vary** the following parameters using the `hyperparameters.yaml`:
- number of `features` for each level and the number of levels
- `stride` and `kernel_size` for down- and up-path and for the `bottleneck`

## Monitoring Unet

If your runs are managed using `hydra`, the results can by nicely monitored
using [`tensorboard`](https://www.tensorflow.org/tensorboard/get_started).
There's an example pytho file using hydra in the example directory of this repo: `example/using_hydra.py`.

For that, navigate to the respective directory and run
    tensorboar --logdir 
