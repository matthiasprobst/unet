# Implementation of a U-NET

Implemetation of the U-NET Structure. Built and adjusted based on https://www.youtube.com/watch?v=IHq1t7NxS8k.


## Monitoring Unet

If your runs are managed using `hydra`, the results can by nicely monitored
using [`tensorboard`](https://www.tensorflow.org/tensorboard/get_started)

For that, navigate to the respective directory and run
    tensorboar --logdir 