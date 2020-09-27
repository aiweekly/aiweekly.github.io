---
title: week 1 - the hardware lottery and Numpy | improving transformer-based models | not-so-BigGAN and BYOL
permalink: /posts/week-1-the-hardware-lottery-and-numpy-and-more
---
This is the first edition of ai weekly where I aim to share my notes on the research that I've found interesting in the past week. This week we will explore 6 impressive papers showcasing the latest innovations in natural language processing and computer vision as well as 2 papers focused on hardware and software efforts and limitations in scientific computing.

### The Hardware Lottery

[https://arxiv.org/abs/2009.06489](https://arxiv.org/abs/2009.06489)

[https://hardwarelottery.github.io/](https://hardwarelottery.github.io/)

**What is it?** This paper argues that throughout computer science history (and even today) available hardware and software have played an important role in determining which research ideas/algorithms succeed and which ones fail.

**Why it matters?** Large research labs are increasing focusing on throwing larger and larger models and more compute at a problem with OpenAI's GPT-3 model being the most recent example on this line of work, costing [around $12 million](https://venturebeat.com/2020/06/01/ai-machine-learning-openai-gpt-3-size-isnt-everything/) to train. There are definitely benefits to running such large experiments to further optimize current algorithms on available hardware, but it is unclear how sustainable such efforts are. While large and well-funded research labs can afford to run costly experiments using available tools, such large experiments are beyond the reach of small AI companies and university research labs. This can kill other ideas and algorithms that could come out of these smaller companies and institutions while all the focus is on million-dollar projects that further optimize current methods that are fully compatible with available tools.

### Array programming with NumPy

[https://www.nature.com/articles/s41586-020-2649-2](https://www.nature.com/articles/s41586-020-2649-2)

**What is it?** This paper provides a comprehensive overview on Numpy which is one of the most foundational building blocks of the scientific Python ecosystem.

**Why it matters?** Numpy is at the core of numerous scientific libraries and projects that depend on its reliability, speed, and stability. Therefore, it is vital for the scientific community to have a good understanding of this library and its future roadmap. More importantly, we need the scientific community to come together to drive the Numpy library forward through the rapidly evolving landscape of scientific computing.

### Pay Attention when Required

[https://arxiv.org/abs/2009.04534](https://arxiv.org/abs/2009.04534)

[https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling)

**What is it?** This paper uses neural architecture search to explore various combinations and proportions of self-attention and feed-forward layers (the building blocks of transformer-based models) and finds more efficient transformer architectures while retaining the performance of the original model.

**Why it matters?** Running transformer-based models on client devices and limited-memory and limited-compute resources is challenging. While techniques such as [quantization and distillation](https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7) can be used to overcome these performance challenges in production, it is also fruitful to investigate redesigning available transformer architectures to better understand how each component contributes to model accuracy and performance and whether a simple and more efficient architecture can provide similar accuracies while dramatically reducing memory footprint and accelerating performance.

### Unit Test Case Generation with Transformers

[https://arxiv.org/abs/2009.05617](https://arxiv.org/abs/2009.05617)

[https://github.com/microsoft/methods2test](https://github.com/microsoft/methods2test)

**What is it?** This paper introduces a transformer-based sequence-to-sequence model for automatically generating unit test cases in Java. It also introduces a parallel dataset of methods and their corresponding unit tests in Java scraped from open-source Github repositories which was used to train their model.

**Why it matters?** Unit testing is a fundamental phase in the software development lifecycle which tends to be challenging and time-consuming for developers. Automating software testing allows engineering teams to focus on developing their software and only spend a fraction of their time on testing while maintaining test quality and correctness. Current automatic unit testing approaches usually generate test cases that are difficult to read and understand for developers. Recent breakthroughs in sequence-to-sequence modeling using transformer-based architectures make them suitable candidates for building translation models that encode functions/methods and automatically generate test cases that are highly readable, understandable, and effective.

### not-so-BigGAN: Generating High-Fidelity Images on a Small Compute Budget

[https://arxiv.org/abs/2009.04433](https://arxiv.org/abs/2009.04433)

[https://github.com/hanseungwook/not-so-biggan-decoder](https://github.com/hanseungwook/not-so-biggan-decoder)

**What is it?** This paper uses wavelet transforms to recursively slice an image into smaller patches representing the entire image along different frequency bands. They then train a small generative model to generate the lowest-frequency patch which is then recursively upsampled by recovering the remaining high-frequency patches using their partly-learned, partly-deterministic decoder. This work is closely related to the [Subscale Pixel Network](https://towardsdatascience.com/generating-high-resolution-images-using-autoregressive-models-3683f9af0db4) architecture.

**Why it matters?** Generative models typically operate in the pixel space making high-resolution image generation a computationally expensive task. And for the past few years, most breakthroughs in large-scale generative modeling have come from research labs within large tech companies while leaving out the wider research community who may not have access to massive computing resources and hefty budgets. The ability to train and use generative models using limited resources will democratize access to the latest and greatest advancements in this area. This will, in turn, inspire new algorithms in areas such as image generation and self-supervised image representation learning among others allowing the wider research community to participate in this line of research. Finally focusing only on large models comes with other negative side-effects such as significant environmental impacts.

### Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning

[https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733)

[https://github.com/lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch)

**What is it?** This paper introduces an algorithm for learning image representations in a self-supervised manner without relying on negative examples. The architecture borrows ideas from reinforcement learning by using a target network and an online network that learns to mimic the target network's image representations. The online network parameters are learned while the parameters of the target network are an exponential moving average of the online network parameters at each training step.

**Why it matters?** The massive computational requirements of generative methods among other challenges make discriminative approaches more practical for self-supervised representation learning in vision with contrastive methods currently achieving impressive performance on various vision benchmarks. This work achieves state-of-the-art results while avoiding some drawbacks of contrastive approaches. Although this work still requires relatively large computational resources for training, it is a step in the right direction that will inspire more efficient representation learning algorithms with higher performance.

### Compositional and Lexical Semantics in RoBERTa, BERT and DistilBERT: A Case Study on CoQA

[https://arxiv.org/abs/2009.08257](https://arxiv.org/abs/2009.08257)

**What is it?** This paper evaluates three pre-trained transformer-based models on certain linguistic capabilities by comparing their performance on a set of tasks (as a gauge for the linguistic phenomena of interest) with and without additional training on similar classification tasks. This additional supervised training is expected to feed the models the missing linguistic knowledge. These three models are 1) BERT the original transformer-based model, 2) RoBERTa an optimized version of BERT using 10x larger training data, and 3) DistilBERT a smaller version of BERT built using knowledge distillation.

**Why it matters?** We need a clear picture of what type of knowledge these transformer-based models transfer to downstream tasks. This will allow us to have a better understanding of the shortcomings of these models and confidently roll out solutions fine-tuned on them in production. This work sheds additional light on the type of linguistic knowledge missing from contextualized word embeddings (outputs of transformer-based models) helping better understand their shortcomings. This could also inspire further investigations into novel architectures to capture a wider set of linguistic phenomena through unsupervised pre-training.

### NeuralQA: A Usable Library for Question Answering (Contextual Query Expansion + BERT) on Large Datasets

[https://arxiv.org/abs/2007.15211](https://arxiv.org/abs/2007.15211)

[https://github.com/victordibia/neuralqa](https://github.com/victordibia/neuralqa)

**What is it?** This paper introduces an easy-to-set-up library for debugging question answering systems. It aims to answer questions such as why a set of documents were retrieved and why a particular answer was selected from a document.

**Why it matters?** Debugging blackbox models is a challenging task and it requires building appropriate tools to help explain the behavior of such models. These tools should easily integrate with existing models/systems to seamlessly explore model behavior. This is important in both research and practice as it would inform research directions and it would build confidence for using these models in practice. Exploring such tools is extra important for understand NLP models as these models are not only difficult to debug but in some cases are difficult to evaluate as well for example in natural language understanding (NLU) and natural language generation (NLG) tasks.