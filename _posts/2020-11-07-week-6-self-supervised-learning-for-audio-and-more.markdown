---
title: week 6 - self-supervised learning for audio | modified attention with performers and QDS | LM for drug discovery and knowledge graphs
permalink: /posts/week-6-self-supervised-learning-for-audio-and-more
---
Welcome to the sixth edition of ai weekly. This week we will explore one paper on applying contrastive learning to audio signals, 2 papers on simplifying the attention mechanism in transformer architectures, and finally 2 papers on applying transformer-based language models to other areas such as predicting molecular properties and building knowledge graphs.

### Contrastive Learning of General-Purpose Audio Representations

[https://arxiv.org/abs/2010.10915](https://arxiv.org/abs/2010.10915)

[https://github.com/google-research/google-research/tree/master/cola](https://github.com/google-research/google-research/tree/master/cola)

**What is it?** This paper introduces a self-supervised objective inspired by contrastive learning methods in computer vision and reinforcement learning for learning latent embeddings for audio signals that can be further fine-tuned and used in various downstream audio classification tasks.

**Why it matters?** Contrastive learning approaches have been successfully used in reinforcement learning and computer vision for learning a latent space in a self-supervised manner. This contrastive objective can be a useful alternative to the triplet loss for extracting features from unlabeled audio data as well. This allows learning audio embeddings that can be used not only for downstream speech tasks but also for other audio tasks such as acoustic scene detection or animal vocalizations. Contrastive learning is also very well aligned with audio signals as you can simply use multiple subsets of an audio clip as positive examples and segments of other clips as negative examples without requiring any augmentation procedure. This work introduces a useful baseline for future work in self-supervised learning for audio.

### Rethinking Attention with Performers

[https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794)

[https://github.com/google-research/google-research/tree/master/performer/fast_self_attention](https://github.com/google-research/google-research/tree/master/performer/fast_self_attention)

**What is it?** This paper introduces a mechanism for approximating the softmax function and estimating the attention matrix in a transformer architecture in linear space and time complexity. This mechanism (FAVOR+) does not assume any priors and provides an unbiased estimation of the attention matrix. This linear transformer architecture (Performer) is also fully compatible with regular transformers.

**Why it matters?** While transformer models have revolutionized several areas of machine learning, their attention matrix calculation remains a computational bottleneck as it scales quadratically with the number of tokens in the input sequence. This work aims to solve this issue by introducing an approach with strong mathematical foundations without restricting or simplifying the attention mechanism. This would allow transformer models to easily scale up and process long input sequences especially in settings with limited computational resources.

### Long Document Ranking with Query-Directed Sparse Transformer

[https://arxiv.org/abs/2010.12683](https://arxiv.org/abs/2010.12683)

[https://github.com/hallogameboy/QDS-Transformer](https://github.com/hallogameboy/QDS-Transformer)

**What is it?** This paper presents a method for ranking long documents using pre-trained transformers by avoiding unnecessary connections between distant document tokens in the attention matrix and only including informative connections as sparse adjacency matrices.

**Why it matters?** While pre-trained transformer-based language models such as BERT have been enormously successful in various NLP tasks, the quadratic complexity of their self-attention operation makes it impossible to apply these models to long documents. This works aims to overcome this issue by designing a sparse attention matrix based on information retrieval principles. This work can inspire future research into incorporating other forms of inductive biases to improve the performance of pre-trained language models on downstream NLP tasks.

### ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction

[https://arxiv.org/abs/2010.09885](https://arxiv.org/abs/2010.09885)

**What is it?** This paper introduces a transformer-based model similar to RoBERTa and trains it on a masked language-modeling (MLM) task to learn a representational topology of chemical space and systematically evaluates the viability of pre-trained transformers on molecular property prediction tasks.

**Why it matters?** Machine learning has the potential to automate and accelerate drug discovery which is crucial for the rapid development of new and life-saving medicine. This work aims to expand the success of transformer-based language models in NLP tasks to a similar sequence prediction task in the chemical space. This method can further be combined with graph-based approaches for learning molecular structures. This work also inspires research into other more recent transformer-based models such as [ELECTRA](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html) for predicting chemical properties.

### Language Models are Open Knowledge Graphs

[https://arxiv.org/abs/2010.11967](https://arxiv.org/abs/2010.11967)

**What is it?** This paper introduces a method for automatically extracting facts from a textual corpus and mapping them to a knowledge graph by leverages the attention matrix from a pre-trained language model to extract the relation between noun phrases in a sentence.

**Why it matters?** This work builds a knowledge graph with a single forward pass of a pre-trained language model over a textual corpus with no training involved. While training transformer-based language models typically require access to massive computational resources, running a forward pass of these models can be done even with limited resources. Algorithms that do not rely on training (or fine-tuning) transformer models, such as the one introduced in this work, can inspire further research into using transformer-based language models without access to large resources. This work also helps researchers explicitly understand the knowledge stored in pre-trained language models weights.