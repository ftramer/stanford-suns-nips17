# Non-targeted Attack

This is code for a non-targeted adversarial attack.  The attacker is allowed to change each pixel of the original image up to some maximum l_inf bound. The aim of this attack is to adversarially direct the image recognition model to predict a wrong label.  

## High-level Strategy

Our strategy is described as follow: we import a family of pre-trained models, and partition them into two categories. One is called the *Blackbox models* and the other is called *Whitebox models*. The Blackbox models are those that we found attacking them generalizes well to unseen models. The Whitebox models are the complement set of Blackbox models. We generate two attacks, for the Blackbox family and the Whitebox family respectively, and obtain the final attack by computing a weighted linear combination of them.

Given an image and a family of models to attack, we first estimate the groundtruth label of this image using a high-precision model (InceptionResNetV2). Then we define a loss function which encourages the pixels to be perturbed such that all models in the family outputs a different label from the groundtruth. We generate the attack by minimizing the loss function. The minimization problem is solved by either Sign Gradient Descent or RMSProp.

## Loss Function

The loss function to minimize is defined as follows.  Here `label_mask` is the one-hot encoded groundtruth label.

```
softmax_prob_sum = 0
for model in models:
    softmax_prob_sum += tf.reduce_sum(tf.nn.softmax(model.logits) * label_mask, axis=1)
self.loss = tf.reduce_mean(tf.log(margin + softmax_prob_sum))
```

To understand this loss, notice that `tf.reduce_sum(tf.nn.softmax(model.logits) * label_mask, axis=1)` is the probaiblity that `model` makes a correct prediction. As a result, `softmax_prob_sum` is the sum of all such probabilities for all models in the family. Minimizing this loss forces all models to have a small success probability. The `margin` is a small positive hyper-parameter that prevents over-fitting and improves generalizability of the attack.

## Average Pooling Layer

We found that for generating attacks, adding an average pooling layer between the image and the pretrained models improves the attackers' generalizability to attack unseen models, especially when the unseen model uses image pre-processing techniques (such as smoothing and JPEG compression). As a result, we defined two models called `SmoothInceptionV3Model` and `SmoothInceptionResNetV2Model`, and added them to the Blackbox family.

## Choosing Models

To choose models in the Blackbox familiy and the Whitebox family, edit the `--blackbox-train` and `--whichbox-train` parameters in `run_attack.sh`.  The model indices correspond to the `all_models` list in `attack.py`.

## Quick Setup

1. To download all model checkpoints, navigate to `model_ckpts` and run:
```bash
./download_all.sh
```
2. In the master directory, download images by running `./download_data.sh`. Images should be stored in `dataset/images/`.

3. Setup an output directory where adversarial images will be stored (e.g. `OUTPUT_DIR`). 

4. To generate adversarial images, run:
```bash
./run_attack.sh ../dataset/images/ OUTPUT_DIR 16
``` 
This will generate adversarial images within an l_inf perturbation of size 16 and store them in `OUTPUT_DIR` with the same filename as the original image. 

 
