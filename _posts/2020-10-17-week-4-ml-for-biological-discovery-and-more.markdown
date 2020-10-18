---
title: week 4 - ML for biological discovery | engaging stories and dialogues | vokenization and contrastive distillation
permalink: /posts/week-4-ml-for-biological-discovery-and-more
---
Welcome to the fourth edition of ai weekly. This week, we will explore ML for experimental design in biological studies, novel frameworks for generating consistent stories and dialogues, exposing language models to data in multiple modalities, and a contrastive distillation technique to reduce the memory and computation footprint of large language models.

### Leveraging Uncertainty in Machine Learning Accelerates Biological Discovery and Design

[https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30364-1](https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30364-1)

**What is it?** This paper shows how biological discovery can be accelerated by using machine learning approaches such as gaussian process-based algorithms that can quantify prediction uncertainty allowing researchers to design experiments around hypotheses with a high likelihood of success. 

**Why it matters?** Researchers in biological settings usually run thousands or millions of experiments per day making it difficult for human experts to evaluate the results and make educated hypotheses that would guide further experimental design. While machine learning algorithms are ideal candidates for automating this hypothesis generation process, their behavior can become unpredictable when exploring regions beyond the training data distribution. This work shows how machine learning methods that quantify prediction uncertainty can overcome this issue and help increase the adoption of these algorithms into biological discovery and experimental design.

### MEGATRON-CNTRL: Controllable Story Generation with External Knowledge Using Large-Scale Language Models

[https://arxiv.org/abs/2010.00840v1](https://arxiv.org/abs/2010.00840v1)

[https://developer.nvidia.com/blog/adding-external-knowledge-and-controllability-to-language-models-with-megatron-cntrl/](https://developer.nvidia.com/blog/adding-external-knowledge-and-controllability-to-language-models-with-megatron-cntrl/)

**What is it?** This paper introduces a framework for generating fluent and consistent stories by leveraging external knowledge sources. At each step, a conditional language model generates the next sentence using the current story context and top external knowledge sentences retrieved based on keywords extracted from the current story context.

**Why it matters?** One of the downsides of large-scale transformer-based language models is their lack of controllability and consistency with real-world facts. The framework introduced here forces a language model to pay more attention to the current context by reinforcing the important words in the context. This is done by combining the context with relevant sentences sourced from an external knowledge-base. This can improve both controllability and fluency as a modification in the context (or keywords extracted from the context) will be reflected in the sentences collected from an external knowledge-base which directly affects the outputs of the language model i.e. the following sentence in the story.

### Like hiking? You probably *enjoy nature*: Persona-grounded Dialog with Commonsense Expansions

[https://arxiv.org/abs/2010.03205](https://arxiv.org/abs/2010.03205)

[https://github.com/majumderb/compac](https://github.com/majumderb/compac)

**What is it?** This paper uses available commonsense knowledge bases (e.g. ConceptNet and ATOMIC) and paraphrasing techniques to generate expansions from an original persona sentence. This method aims to capture the additional implications of a. persona sentence such as 'I love surfing' and infer that the person is 'adventurous' or ' loves the outdoors' or 'enjoys going to the beach'. This expanded set of persona sentences allows a dialogue model to generate more engaging and context-consistent responses.

**Why it matters?** Automated dialogue systems can only gain popularity and become commonplace if they are engaging for users. Previous research has shown that these models can become more appealing when they have personalized back-stories. This work creates engaging back-stories for these systems by conditioning a dialogue generation model on a rich set of inferred persona sentences in addition to the conversation history.

### Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision

[https://arxiv.org/abs/2010.06775](https://arxiv.org/abs/2010.06775)

[https://github.com/airsplay/vokenization](https://github.com/airsplay/vokenization)

**What is it?** This paper develops a method for generating token-related images or vokens (visualized tokens) and uses them to train visually-supervised language models that show improvements on various NLP tasks when compared to language models trained on pure language corpora

**Why it matters?** While large transformer-based language models have gained substantial improvements on various NLP benchmarks, they are all trained on massive corpora of text which is very different from how humans understand language. Humans learn language not by reading lots of text but by interacting with the real world through multiple modalities. Further progress towards understanding human language would only be possible by going beyond text and grounding information from multiple modalities.

### Contrastive Distillation on Intermediate Representations for Language Model Compression

[https://arxiv.org/abs/2009.14167](https://arxiv.org/abs/2009.14167)

[https://github.com/intersun/CoDIR](https://github.com/intersun/CoDIR)

**What is it?** This paper introduces an effective distillation technique for transformer architectures by using a contrastive loss between the intermediate layers of the teacher network and the student network. The student network takes in one positive sample and multiple negative samples and it aims to increase the distance between its intermediate representations of the negative samples with the teacher network's intermediate representation of the positive example while keeping the two networks' representation of the positive example close to each other.

**Why it matters?** Large-scale language models are power-hungry and rely on massive amounts of data for pre-training making them impractical in low-resource settings. Robust distillation techniques can overcome this issue by teaching a smaller student network to capture the knowledge stored in the weights of the teacher network. Current distillation methods use the mean squared error between the weights in the intermediate layers of the teacher and student networks. While simple this loss function does not allow the student network to explore the rich information in the teacher network's hidden layers. This works aims to solve this issue by using a contrastive loss instead.