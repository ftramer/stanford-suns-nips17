#!/usr/bin/env bash

# Inception V3
if [ ! -f inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
fi

# Inception V4
if [ ! -f inception_v4.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
    tar -xvf inception_v4_2016_09_09.tar.gz
    rm inception_v4_2016_09_09.tar.gz
fi

#ResNet V1 50
if [ ! -f resnet_v1_50.ckpt ]; then
    wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    tar -xvf resnet_v1_50_2016_08_28.tar.gz
    rm resnet_v1_50_2016_08_28.tar.gz
fi

# ResNet V2 50 
if [ ! -f resnet_v2_50.ckpt ]; then
    wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
    tar -xvf resnet_v2_50_2017_04_14.tar.gz
    rm resnet_v2_50_2017_04_14.tar.gz
fi

# ResNet V2 101
#if [ ! -f resnet_v2_101.ckpt ]; then
#    wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
#    tar -xvf resnet_v2_101_2017_04_14.tar.gz
#    rm resnet_v2_101_2017_04_14.tar.gz
#fi


# MobileNet
#if [ ! -f mobilenet_v1_1.0_224.ckpt.meta ]; then
#    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
#    tar -xvf mobilenet_v1_1.0_224_2017_06_14.tar.gz
#    rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
#fi

# Inception ResNet V2
if [ ! -f inception_resnet_v2_2016_08_30.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    tar -xvf inception_resnet_v2_2016_08_30.tar.gz
    rm inception_resnet_v2_2016_08_30.tar.gz
fi

# VGG
if [ ! -f vgg_16.ckpt ]; then
    wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    tar -xvf vgg_16_2016_08_28.tar.gz
    rm vgg_16_2016_08_28.tar.gz
fi

# Adversarially trained Inception V3
if [ ! -f adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvf adv_inception_v3_2017_08_18.tar.gz
    rm adv_inception_v3_2017_08_18.tar.gz
fi

# Ensemble adversarially trained Inception V3
if [ ! -f ens3_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens3_adv_inception_v3_2017_08_18.tar.gz
    rm ens3_adv_inception_v3_2017_08_18.tar.gz
fi

# Ensemble adversarially trained Inception V4
if [ ! -f ens4_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens4_adv_inception_v3_2017_08_18.tar.gz
    rm ens4_adv_inception_v3_2017_08_18.tar.gz
fi

# Ensemble adversarially trained Inception Resnet v2
if [ ! -f ens_adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi

# Keras Xception model
if [ ! -f keras_xception.pb ]; then
    cp /scr/zhangyuc/stanford-nips17-competition/dataset/keras_xception.pb ./
fi
