---
layout: post
comments: true
title: "Image similarity: one-shot embedding"
excerpt: "I'll discuss ..."
date:   2020-11-20 01:00:00
mathjax: true
---


## Introduction
Image similarity is an important building block in many machine learning systems, such as recommendation system, image retrieval system. However, depending on the applications, the semantic definition of similarity varies a lot. Below is an illustration of different notions of similarity from [Veit 2017](https://vision.cornell.edu/se3/wp-content/uploads/2017/04/CSN_CVPR-1.pdf): 


<div class="imgcap">
<img src="/assets/embedding_cnn/notions.png" height="300">
<div class="thecap">notions.</div>
</div>

To measure the similarity between images, images are typically embedded in a feature-vector space, where the distances, such as cosine distance, between the vectors are derived as the relative dissimilarity.

		The image embedding can be derived from both supervised and un-supervised fashion. In the supervised approach, either the pairwise [Chopra 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) or triplet [Wang 2014](https://arxiv.org/pdf/1404.4661.pdf) loss function are chosen to supervise the learning, which requires a large amount of labelled data set or a well-designed way to generate training data. On the other hand, the un-supervised approach typically involves certain pre-trained neural network architecture, where the last layer is replaced by the embedding layer, to carry out one-shot learning.

In this article, we will illustrate the un-supervised approach to derive the image embedding and calculate the cosine similarity on two publicly available fashion data sets. The implementation will be done with `pytorch`.

## Methods
### Data set
[UT Zappos50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/)
> UT Zappos50K (UT-Zap50K) is a large shoe dataset consisting of 50,025 catalog images collected from Zappos.com. The images are divided into 4 major categories — shoes, sandals, slippers, and boots — followed by functional types and individual brands. The shoes are centered on a white background and pictured in the same orientation for convenient analysis.


[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
> We contribute DeepFashion database, a large-scale clothes database, which has several appealing properties:
> 
> - DeepFashion contains over 800,000 diverse fashion images ranging from well-posed shop images to unconstrained consumer photos. 
> 
> - DeepFashion is annotated with rich information of clothing items. Each image in this dataset is labeled with 50 categories, 1,000 descriptive attributes, bounding box and clothing landmarks.
>
> - DeepFashion contains over 300,000 cross-pose/cross-domain image pairs.


### Pre-trained model
- [RESNET](https://pytorch.org/hub/pytorch_vision_resnet/) [[Original paper](https://arxiv.org/pdf/1512.03385.pdf)]

### Source code
The source code in this article can be found [image similarity](https://gitlab.com/abinitio/image_similarity/vanilla_cnn).



## Image embedding



## Similarity metrics






## Summary


