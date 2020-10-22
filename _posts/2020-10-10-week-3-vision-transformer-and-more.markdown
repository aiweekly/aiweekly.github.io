---
title: week 3 - vision transformer | EvolGAN and G-SimCLR | stylistic text generation and dense retrieval
permalink: /posts/week-3-vision-transformer-and-more
---
Welcome to the third edition of ai weekly. This week, we have three computer vision papers including a CNN-free/transformer-only network, an evolutionary GAN framework for improving image quality, and an improved contrastive learning method for self-supervised learning. We also have two natural language processing papers focused on conditional generation of fluent text with varying styles and a dense information retrieval model.

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

[https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)

**What is it?** This paper breaks down an image into 16 by 16 pixel patches and feeds this sequence of patches to a transformer to perform image classification without any convolutional layers.

**Why it matters?** While convolutional architectures have become the standard in computer vision, evidently a standalone transformer architecture can be applied to vision tasks almost out of the box to outperform the state-of-the-art convolutional networks. Convolutional architectures have been useful as an inductive bias when using smaller datasets, but now that we have enough data and compute power we can go beyond the manual heuristics of convolutional networks and use the more generic transformer architecture to let the model learn those heuristics.

### EvolGAN: Evolutionary Generative Adversarial Networks

[https://arxiv.org/abs/2009.13311](https://arxiv.org/abs/2009.13311)

**What is it?** This paper introduces an evolutionary optimization method for optimizing the latent code passed to the generator. This allows the generator to map this input to an image with significantly higher quality (in certain classes) without modifying the weights of the GAN network.

**Why it matters?** A GAN model can synthesize high-quality images once trained on a massive dataset that is carefully curated. This work allows a GAN model to be compatible with small datasets and continue to generate impressive results even in the absence of huge datasets and without modifying the training procedure. This optimization approach also has a reasonable computational cost as it improve the quality of the generated images without retraining the original GAN model.

### G-SimCLR: Self-Supervised Contrastive Learning with Guided Projection via Pseudo Labelling

[https://arxiv.org/abs/2009.12007](https://arxiv.org/abs/2009.12007)

[https://github.com/ariG23498/G-SimCLR](https://github.com/ariG23498/G-SimCLR)

**What is it?** This paper minimizes the risk of including images of the same class in a training batch (as positive-negative pairs) by clustering the latent representations of unlabelled images and essentially pseudo labelling the data before running self-supervised training.

**Why it matter?** While supervised learning methods can achieve impressive results, they require massive amounts of labelled data for training which is extremely expensive and time-consuming to collect and annotate. This work improves the quality of representations learned by [SimCLR](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html) (a self-supervised contrastive learning method) by avoiding images of the same class as positive-negative pairs in a training batch. Without this constraint on the mini-batches the original SimCLR maximizes the distance between images of the same class and forces the model to distance similar images in the latent space which would hurt performance. This can also inspire future work for resolving other known issues of contractive learning methods.

### Controllable Text Generation with Focused Variation

[https://arxiv.org/abs/2009.12046](https://arxiv.org/abs/2009.12046)

**What is it?** This paper learns disjoint and discrete latent codes for the style and content of texts allowing control over the style (control attribute) while generating diverse and fluent text.

**What it matters?** Current conditional language models fail to convey the entire content of an input text nor do they preserve the diversity found in training examples when controlling for an attribute such as style in generated outputs. Natural-sounding text generation is an essential component of systems that interact with humans through natural language and this is only possible if responses are stylistically consistent while the content is coherent and diverse. This paper can inspire future research on improving the quality of generated text when controlled on other predetermined attributes such as emotion.

### Dense Passage Retrieval for Open-Domain Question Answering

[https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

[https://github.com/facebookresearch/DPR](https://github.com/facebookresearch/DPR)

**What is it?** This paper trains a Siamese network to learn dense representations of passages and questions and uses a similarity measure on these representations to find the most relevant passages to an input question.

**Why it matters?** Open-domain QA systems require an information retrieval component to focus on relevant context for responding to questions. This retrieval component has traditionally been a sparse vector search model such as TF-IDF and BM25 that cannot match seemingly different tokens that are semantically identical. This shortcoming has inspired the use of dense vector representations which allow semantically similar questions and passages to be mapped close to each other for efficient retrieval of relevant passages during inference using similarity search methods. This work introduces a simple yet effective dense retrieval system by building a one-time dense index for passages without re-indexing them during training, avoiding complex training schemes.