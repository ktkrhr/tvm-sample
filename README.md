# TVM Execution Sample

## Prerequisite

[TVM installation](https://docs.tvm.ai/install/from_source.html)

For quick trying, [Official Docker images](https://docs.tvm.ai/install/docker.html) are available.

### Other Python dependencies

After TVM Python package installation, install other packages.

```
$ pip install -r requirements.txt
```

## Keras model

Download Keras MobileNet v1 model(`mobilenet_1_0_224_tf.h5`) from [here](https://github.com/fchollet/deep-learning-models/releases)

## Run

```
$python run_tvm.py cat.jpg

.
.
.

Classification Result:
1 tiger cat 0.439913
2 tabby 0.434570
3 Egyptian cat 0.104559
4 lynx 0.011487
5 tiger 0.003490

Evaluate inference time cost...
Inference time 0: 0.990710
Inference time 1: 0.999410
Inference time 2: 0.995719
Inference time 3: 0.984265
Inference time 4: 0.985593
Inference time 5: 0.986964
Inference time 6: 0.991390
Inference time 7: 0.992035
Inference time 8: 1.000641
Inference time 9: 1.001794
Mean inference time (std dev): 0.992852 ms (0.006003 ms)

```
