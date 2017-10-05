# Targeted Attack

This is code for a targeted adversarial attack.  The attacker is allowed to change each pixel of the original image up to some maximum l_inf bound. The aim of this attack is to adversarially direct the image recognition model to predict towards particular classes.  

Our approach is adapted from the projected gradient descent technique presented in this paper: [Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2017)](https://arxiv.org/pdf/1706.06083.pdf).

## Loss Function

The loss function to minimize is defined as follows.  Here `label_mask` is the one-hot encoded target class.

This targeted attack is broadly adapted from the non-targeted attack code, with the following key difference:

 - The non-targeted attack minimizes the output probability associated with the true class.
 - The targeted attack maximizes the output probability associated with the target class.

```
softmax_prob_sum = 0
for i in train:
    softmax_prob_sum += tf.reduce_sum(tf.nn.softmax(models[i].logits) * label_mask, axis=1)
self.mixture_loss = (-1.0) * tf.reduce_mean(tf.log(margin + softmax_prob_sum))
```

## Ensembling Models

To choose models to ensemble, edit the `--whitebox-train` parameter in `run_attack.sh`.  The model indices correspond to the `all_models` list in `attack.py`.

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

 
