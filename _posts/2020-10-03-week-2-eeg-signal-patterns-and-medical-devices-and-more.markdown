---
title: week 2 - EEG patterns and medical devices | GroC and GeDi | QA and dialogue systems
permalink: /posts/week-2-eeg-signal-patterns-and-medical-devices-and-more
---
Welcome to the second edition of ai weekly. A short read presenting some of the most interesting research on machine learning I've found in the past week. If you missed the first week's issue, you can check it out [here](https://aiweekly.github.io/posts/week-1-the-hardware-lottery-and-numpy-and-more).

### Uncovering the structure of clinical EEG signals with self-supervised learning

[https://arxiv.org/abs/2007.16104](https://arxiv.org/abs/2007.16104)

**What is it?** This paper introduces a number of self-supervised objectives for learning EEG signal representations. Using these learned representations they train linear classifiers on smaller annotated datasets and compare their results with various supervised learning methods. Note that these pre-trained models are not fine-tuned per se (as in language models such as [BERT](https://arxiv.org/abs/1810.04805)) and the weights of these models are frozen after pre-training.

**What it matters?** The vast number of physiological monitoring devices available outside clinical domains generate an ever-increasing amount of physiological data that need to be analyzed. Automating this analysis is challenging as it can be expensive and time-consuming (and sometimes impossible) to annotate large physiological datasets. To harness the full power of data-hungry deep learning algorithms on physiological signals we need new approaches that can be trained on large amounts of unlabeled recordings without relying on expert annotations. Self-supervised approaches are successfully used in domains such as natural language processing and computer vision to build pre-trained models that can be fine-tuned on downstream tasks using smaller labeled datasets. These approaches can be used in biosignals as well to capture physiological insights in the absence of labeled data and this work can inspire future research along this path.

### The state of artificial intelligence-based FDA-approved medical devices and algorithms: an online database

[https://www.nature.com/articles/s41746-020-00324-0](https://www.nature.com/articles/s41746-020-00324-0)

[https://medicalfuturist.com/fda-approved-ai-based-algorithms/](https://medicalfuturist.com/fda-approved-ai-based-algorithms/)

**What is it?** This paper provides an overview of AI/ML-based medical solutions currently approved by the US FDA. They compile their findings into an open-access database that is to be continuously maintained and updated.

**Why it matters?** Some companies in the healthcare space tend to use AI as a PR move to attract further investments by claiming to have an AI/ML-based technology while in reality, they may not be using any ML algorithms per se. It is vital for regulatory bodies such as the FDA to separate actual AI/ML-based technologies from the systems that are hyped-up with buzz words and devise a clear evaluation process for such technologies as they are fundamentally different from rule-based systems making their approval process more delicate. The power of these AI/ML-based algorithms lies within their probabilistic nature as the probabilities they assign to various outcomes can change over time by adapting to new signals from real-world experience. This characteristic while powerful makes the review process of such systems extremely complicated. This work paints a realistic picture of the current state of AI/ML-based solutions and describes the current weaknesses of their approval system which can in turn accelerate the search for solutions to these shortcomings.

### Grounded Compositional Outputs for Adaptive Language Modeling

[https://arxiv.org/abs/2009.11523](https://arxiv.org/abs/2009.11523)

[https://github.com/Noahs-ARK/groc](https://github.com/Noahs-ARK/groc)

**What is it?** This paper decouples pre-trained language model training vocabulary from the target vocabulary in a downstream task during inference or fine-tuning making the model parameters completely independent of vocabulary size. In particular, they use a compositional input embedding grounded in information from an external structured lexicon ([WordNet](https://wordnet.princeton.edu/)).

**Why it matters?** Word-based language models tend to be confined to their training vocabulary. One solution to this is to use subword tokenization strategies such as Byte-Pair Encoding (BPE), WordPiece, and SentencePiece. While these approaches solve the out of vocabulary issue, they require larger model capacities and are slower to converge. The adaptive language model introduced in this work opens up the vocabulary while reducing model parameters and increasing sample efficiency. The adaptive nature of this word-based language model is particularly useful for extending pre-trained language models to specialized, low-resource domains where the model has not seen a large portion of the vocabulary during training.

### GeDi: Generative Discriminator guided Sequence Generation

[https://arxiv.org/abs/2009.06367](https://arxiv.org/abs/2009.06367)

[https://github.com/salesforce/GeDi](https://github.com/salesforce/GeDi)

**What is it?** This paper splits class conditional language modeling into two separate components by training a smaller conditional language model (in this case the smallest version of GTP-2) as a discriminator to efficiently guide the predictions of a larger vanilla language model (they use the largest GPT-2 here). At each time step the probability predictions of the discriminator for a desired control token (e.g. positive) are multiplied by the raw probabilities of the vanilla language model to guide generation towards the desired attribute (e.g. positive).

**Why it matters?** Conditional language models usually struggle with unseen/out-of-domain data as well as unseen control tokens as they tend to be biased towards in-domain data and the conditional tokens used during training. This work combines ideas from previous research on conditional language modeling to propose an efficient model that generalizes to new domains and control codes. This work can inspire future research on improving text generation systems in a computationally efficient manner without resorting to larger models.

### QED: A Framework and Dataset for Explanations in Question Answering

[https://arxiv.org/abs/2009.06354](https://arxiv.org/abs/2009.06354)

[https://github.com/google-research-datasets/QED](https://github.com/google-research-datasets/QED)

**What is it?** This paper introduces a framework and dataset for building explainable and transparent question answering (QA) systems along with baseline models on two tasks. This framework decomposes the QA process into 4 human-interpretable subproblems as follows 1) single sentence selection 2) single answer selection 3) identification of question-sentence noun phrase equalities and 4) extraction of an entailment pattern

**Why it matters?** AI/ML system explainability is vital in both research and practice as it helps practitioners understand and debug these systems while it helps researchers extend the models. It also helps users and operators of AI systems build confidence and trust in them. Finally, it allows external evaluators to effectively assess these systems without any technical knowledge of the inner-workings of such models.

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

**What is it?** This paper combines a pre-trained neural retriever (query encoder + document index which is the non-parametric knowledge memory) for accessing Wikipedia articles with a pre-trained sequence-to-sequence model (the parametric knowledge memory) for response generation and tackles a variety of knowledge-intensive NLP tasks.

**Why it matters?** While pre-trained language models can store factual knowledge in their parameters and can generate impressive factual text, their ability is limited in knowledge-intensive NLP tasks. Moreover, since this knowledge is directly stored in model weights, factual knowledge can only be updated by retraining these models which is inefficient. This work goes beyond stand-alone language models by combining a language model with a transparent non-parametric knowledge store. The non-parametric part is not a mere extractive model rather it allows the language model to condition its generations on relevant factual knowledge and generate factually correct responses. Furthermore, since the non-parametric knowledge store is decoupled from the language generation model, its knowledge can be updated without having to retrain the model.

### Task-Oriented Dialogue as Dataflow Synthesis

[https://arxiv.org/abs/2009.11423](https://arxiv.org/abs/2009.11423)

[https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis)

**What is it?** This paper introduces a dataflow framework and a dataset for learning a mapping from natural-language dialogues to graphs of function calls.

**Why it matters?** A dialogue system if reliable can decrease the human workload and support a broad range of applications such as in call centers, help desks, drive-thru services, etc. The accuracy of such systems will start to rapidly improve once well-founded frameworks and datasets are available to automatically train such systems. While end-to-end neural sequence-to-sequence models have shown promise in directly mapping conversational histories to API calls (without a representation of the dialogue state), they tend to fall behind rule-based baselines. This work shows that mapping user queries to a graph of programs/functions (i.e. a dataflow representation as described in the paper) would efficiently track the dialogue state allowing an off-the-shelf sequence-to-sequence model to effectively translate user natural-language queries to machine-understandable actions (i.e. API calls).