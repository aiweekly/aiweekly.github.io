---
title: week 5 - NeuralDB and LaND | speech separation and semi-supervised ASR | deepfake detection and artistic styleGAN
permalink: /posts/week-5-neuraldb-and-land-and-more
---
Welcome to the fifth edition of ai weekly. This week we will explore 6 papers covering a diverse set of topics including neural databases, reinforcement learning, speech separation and localization, speech detection, fake content detection, and image synthesis.

### Neural Databases

[https://arxiv.org/abs/2010.06973](https://arxiv.org/abs/2010.06973) 

**What is it?** This paper represents the facts in a database as a set of natural language sentences with no pre-defined schema where queries are given in natural language as well. This is essentially a more complex version of an open-book QA system.

**Why it matters?** Recent neural network advancements in learning from massive amounts of unstructured data (in natural language understanding, speech recognition, and computer vision) beg the question of whether these same networks could be used to create unstructured database management systems. Is it possible to represent databases and queries as unstructured data with no pre-defined schema and use a neural network to process queries formed using natural language sentences and retrieve the relevant information from a collection of facts? This work attempts to answer this question and inspire future research in this direction to realize the full potential of neural techniques and leverage the recent advancement in this area to create novel neural-based databases.

### LaND: Learning to Navigate from Disengagements

[https://arxiv.org/abs/2010.04689](https://arxiv.org/abs/2010.04689)

[https://github.com/gkahn13/LaND](https://github.com/gkahn13/LaND)

**What is it?** This paper introduces a model for predicting actions that lead to disengagements (i.e. when the system fails and a human most intervene) by leveraging datasets that are naturally collected when testing autonomous systems. A robot can then use this additional learning signal to better navigate complex environments.

**Why it matters?** Autonomous robots are becoming more and more prevalent with recent advancements in delivery robots and autonomous vehicles among other applications. One of the main challenges with developing such systems is the cost of gathering additional training data especially when additional labeled data does not necessarily improve performance. The framework introduced in this work aims to solve this issue by using the disengagement data that is already available from previous experiments to improve performance in future iterations. An autonomous system could also use this additional disengagement information to automatically ask for human intervention when for example it cannot confidently predict whether an action will lead to disengagement or not. This can significantly reduce costs associated with the tedious process of constantly monitoring a robot.

### The Cone of Silence: Speech Separation by Localization

[https://arxiv.org/abs/2010.06007](https://arxiv.org/abs/2010.06007) 

[https://github.com/vivjay30/Cone-of-Silence](https://github.com/vivjay30/Cone-of-Silence)

**What is it?** This paper introduces a network that learns to isolate speech coming from a particular angular region (specified by a target angle and a window size) and disregards speech coming from other directions. The algorithm iteratively decreases the search window size in an angular region, disregarding the regions with no sound at each step until it separates and localizes all sources in logarithmic time.

**Why it matters?** While humans can separate and localize sources of sound, we cannot understand a conversation when multiple people are talking at the same time. However we can build systems that can selectively cancel certain audio sources and only keep the one we are interested in. This capability is especially needed nowadays with an ever-increasing presence of multi-microphone devices such as headphones, laptops, and smart home devices in our daily lives. These devices can potentially use a model similar to the one introduced in this paper to selectively cancel out audio that you don't want to listen to.

### Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition

[https://arxiv.org/abs/2010.10504](https://arxiv.org/abs/2010.10504)

**What is it?** This paper uses both labeled and unlabeled datasets to train an ASR network in a semi-supervised fashion. They fuse a pre-trained [Conformer](https://arxiv.org/abs/2005.08100) encoder with a trained language model to generate labeled data and using this additional data they iteratively fine-tune student models on a mixture of data generated at previous steps.

**Why it matters?** Semi-supervised learning approaches have been effectively used to improve automatic speech recognition (ASR) models where unlabeled data is used in addition to labeled data to further improve model performance. These approaches either use a model trained on the labeled data to generate labels for the unlabeled set and train another model on the entire dataset or they aim to learn the data representation in an unsupervised manner and fine-tune the model using the labeled data. This work aims to combine the best of both worlds by using a fusion of recent advancements in semi-supervised ASR learning to push the state-of-the-art on this task.

### Neural Deepfake Detection with Factual Structure of Text

[https://arxiv.org/abs/2010.07475](https://arxiv.org/abs/2010.07475)

**What is it?** This paper combines word representations with sentence representations to build a document representation which is used to classify a document as either human- or machine-generated. Word embeddings are calculated using RoBERTa and sentence representations are calculated by learning the graph structure of named entities in the document.

**Why it matters?** There is currently an unprecedented amount of misinformation circulating online. This is especially alarming in 2020 with the vast amount of false information on topics such as the COVID-19 pandemic which can cost lives. There have been efforts such as [le GARLIC](https://legarlic.github.io/) that show how easy it is to use models such as GPT-{n} or Grover to generate fake content, and this paper aims to address this issue by introducing a model that can distinguish between machine-generated text and human-written text with improved accuracy.

### Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains

[https://arxiv.org/abs/2010.05334](https://arxiv.org/abs/2010.05334)

[https://github.com/justinpinkney/toonify](https://github.com/justinpinkney/toonify)

**What is it?** This is a very short paper, introducing a simple idea that can generate interesting results. They interpolate between two [styleGAN](https://arxiv.org/abs/1812.04948) models (a pre-trained base model and a model fine-tuned on a new dataset) by using the weights of each model at specific resolution levels, blending features between the two generators.

**Why it matters?** The widespread availability of pre-trained generative models combined with novel ideas such as the one introduced in this work makes generative technologies (e.g. GAN models) more accessible to artists and creatives allowing them to create high-quality creative work without needing access to specialized computing resources. These novel creations can also help form ideas and inspire further progress in generative modeling research.