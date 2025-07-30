---
layout: post
title: "HCMC AI Competition 2025"
date: 2025-07-29 10:00:00 +0700
categories: ai competition
---

# HCMC AI Competition: AI Challenge 2025 – Track 3

## Table of Contents
- [1. Retrieval Task](#retrieval-task)
  - [1.1. What is Retrieval?](#what-is-retrieval)
  - [1.2. Objective](#objective)
  - [1.3. Evaluation Metrics](#evaluation-metrics)
    - [1.3.1. Recall@K (R@K)](#recallk-rk)
    - [1.3.2. Mean Average Precision (mAP)](#mean-average-precision-map)
    - [1.3.3. Mean Rank / Median Rank](#mean-rank--median-rank)
  - [1.4. Example Model Evaluation](#example-model-evaluation)
  - [1.5. Pseudo-Code for Evaluation](#pseudo-code-for-evaluation)
  - [1.6. Pipeline Overview](#pipeline-overview)
  - [1.7. Cosine Similarity](#cosine-similarity)
  - [1.8. Embedding Process](#embedding-process)
  - [1.9. Retrieval Idea](#retrieval-idea)
- [2. TransNetV2](#transnetv2)
  - [2.1. Overview](#overview)
  - [2.2. Architecture](#architecture)
  - [2.3. How It Works](#how-it-works)
  - [2.4. Applications](#applications)
  - [2.5. Resources](#resources)
- [3. CLIP](#clip)
  - [3.1. What is CLIP?](#what-is-clip)
  - [3.2. CLIP Architecture](#clip-architecture)
  - [3.3. Usage Steps](#usage-steps)
  - [3.4. Example](#example)
- [4. FAISS](#faiss)
  - [4.1. What is FAISS?](#what-is-faiss)
  - [4.2. Why Use FAISS?](#why-use-faiss)
  - [4.3. How FAISS Works](#how-faiss-works)
  - [4.4. Basic Example in Python](#basic-example-in-python)

---

# 1. Retrieval Task

## 1.1. What is Retrieval?

Text-Image Retrieval is a task in artificial intelligence within the domain of multimodal information retrieval. The goal is to develop a system capable of:

- **Text-to-Image Retrieval**: Given a textual description, the system retrieves the most relevant images from a dataset.
- **Image-to-Text Retrieval**: Given an image, the system retrieves the most relevant textual descriptions.

In the **AI Challenge 2025 – Track 3**, the focus is on **Text-to-Image Retrieval**. The task involves mapping text and images into a shared embedding space and measuring their similarity to retrieve results.

## 1.2. Objective

Develop a Text-to-Image Retrieval system that:
- Accepts a textual description as input.
- Returns the most relevant images from a dataset.

## 1.3. Evaluation Metrics

### 1.3.1. Recall@K (R@K)

**Definition**: Measures the percentage of queries where the correct image appears in the top K results.

**Formula**:
```
Recall@K = (Number of queries with correct image in top K) / (Total number of queries)
```

**Example**:
- Total queries: 100
- Queries with correct image in top 5: 65
- **Recall@5**: 65%

### 1.3.2. Mean Average Precision (mAP)

**Definition**: The average of the Average Precision (AP) scores across all queries, evaluating both the quality and ranking of results.

**Formula**:
```
mAP = (1/N) × Σ AP_i
```

**AP Calculation**: Average precision is the mean of precisions at positions where correct images are found.

**Example**:
- Correct images at positions 1, 3, 4
- AP = (1/1 + 2/3 + 3/4) / 3 ≈ 0.861

### 1.3.3. Mean Rank / Median Rank

**Definition**:
- **Mean Rank**: The average rank of the first correct image in the result list.
- **Median Rank**: The median rank of the first correct image, more robust to outliers.

**Significance**:
- A lower **Mean Rank** indicates the model retrieves correct images earlier.
- **Median Rank** is more stable with noisy data.

## 1.4. Example Model Evaluation

- **Recall@1**: 0.42
- **Recall@5**: 0.75
- **mAP**: 0.61
- **Median Rank**: 3

## 1.5. Pseudo-Code for Evaluation

```python
def recall_at_k(rank_list, k):
    return sum([1 if rank < k else 0 for rank in rank_list]) / len(rank_list)

def mean_rank(rank_list):
    return sum(rank_list) / len(rank_list)

def median_rank(rank_list):
    return sorted(rank_list)[len(rank_list) // 2]
```

## 1.6. Pipeline Overview

- **Input to Output**: Frames can be extracted from an initial video. A textual description is processed by the retriever to return the corresponding image (frame) matching the description.
  ![alt text](/image/hcmc_ai_competition_2025/{EAC62AC8-9035-44FF-9F9C-BBDD914F5AA7}.png)

- **Retrieval Process**: A user inputs a textual description via an interface (e.g., website), and the system processes it to return the matching image.
  ![alt text](/image/hcmc_ai_competition_2025/{DB2D1E97-DE07-4B01-8D35-98EFB56EA725}.png)
- **Pipeline**:
  ![alt text](/image/hcmc_ai_competition_2025/{8168DCE6-DFD4-407B-A169-AB4261D8535A}.png)

## 1.7. Cosine Similarity

**Definition**: Cosine Similarity measures the similarity between two vectors by calculating the cosine of the angle between them in a vector space. It focuses on the **direction** rather than the magnitude, making it ideal for applications like text-image comparison, recommendation systems, and more.

**Formula**:
```
Cosine Similarity = cos(θ) = (A · B) / (||A|| ||B||)
```

**Output Range**:
- **[-1, 1]**:
  - **1**: Vectors are perfectly aligned (identical).
  - **0**: Vectors are orthogonal (unrelated).
  - **-1**: Vectors are completely opposite.
- In practice, values typically range from **[0, 1]**.

**Example**:
- Vector A = [1, 2, 3]
- Vector B = [2, 4, 6] (B = 2A, same direction)
- Cosine Similarity = **1** (perfectly similar)

**Applications**:
- **Text-Image Retrieval**: Used in CLIP to match text descriptions with images.
- **Recommender Systems**: Suggests items based on user and item vector similarity.
- **NLP**: Compares text, documents, or word/sentence embeddings.
- **FAISS/ANN Search**: Finds nearest vectors in large datasets using cosine similarity.

**Key Insight**:
> Cosine Similarity evaluates the directional similarity of vectors, ignoring their magnitude. This makes it suitable for measuring semantic or behavioral similarity.

- **Cosine Similarity Diagram**:
  ![alt text](/image/hcmc_ai_competition_2025/image.png)

## 1.8. Embedding Process

The embedding step primarily leverages CLIP to map text and images into a shared vector space.

  ![alt text](/image/hcmc_ai_competition_2025/{61B39B56-434D-4D09-88DE-F66EE8DE3706}.png)

## 1.9. Retrieval Idea

The query text is embedded using CLIP, and database images are also embedded. Cosine similarity is computed to retrieve the top-K most relevant results.
  ![alt text](/image/hcmc_ai_competition_2025/{605FAD09-6F3A-45EC-AD81-085EEE80005E}.png)

---

# 2. TransNetV2

## 2.1. Overview

**TransNetV2** is a deep learning model designed for **shot boundary detection**, identifying transitions in videos (e.g., cuts, fades, or soft transitions). It is a critical preprocessing step for tasks like retrieval or highlight extraction. TransNetV2 is an improved version of TransNet, developed by Google Research.

![alt text](/image/hcmc_ai_competition_2025/{4C2182F1-5812-4128-8269-3A65C407E13B}.png)

## 2.2. Architecture

TransNetV2 consists of two main branches:
- **Slow Pathway**: Captures long-term temporal information.
- **Fast Pathway**: Captures short-term temporal information.

The outputs from both pathways are combined to enhance the accuracy of detecting shot boundaries.

## 2.3. How It Works

1. **Input**: A sequence of consecutive video frames.
2. **Feature Extraction**: A CNN extracts features from the frames.
3. **Fusion & Prediction**:
   - Combines short-term and long-term information.
   - Predicts the likelihood of a shot transition at each frame.
4. **Post-Processing**: Filters noise and merges results to pinpoint exact transition locations.

## 2.4. Applications

- **Retrieval**: Segments videos into meaningful clips before embedding, improving retrieval quality.
- **Highlight Detection**: Identifies prominent scenes.
- **Video Summarization**: Automatically selects key segments for summaries.
- **Video Editing**: Assists editors in identifying cut points.

## 2.5. Resources

- **Paper**: [arXiv:2008.04838](https://arxiv.org/abs/2008.04838)
- **GitHub**: [TransNetV2 Repository](https://github.com/soCzech/TransNetV2)
- **Pretrained Models**: Available for easy integration with TensorFlow or OpenCV.

---

# 3. CLIP

## 3.1. What is CLIP?

**CLIP** (Contrastive Language-Image Pretraining) is an OpenAI model that maps images and text into a shared embedding space. It enables:
- Text-to-Image retrieval.
- Image-to-Text retrieval.
- Zero-shot classification.

## 3.2. CLIP Architecture

- **Image Encoder**: Uses ResNet or Vision Transformer (ViT) to process images.
- **Text Encoder**: Uses a Transformer to process text.
- **Training**: Employs contrastive learning to bring matching image-text pairs closer in the embedding space while pushing unrelated pairs apart.

## 3.3. Usage Steps

1. Encode the image using the image encoder → vector \( v_{img} \).
2. Encode the text description using the text encoder → vector \( v_{txt} \).
3. Compute **cosine similarity** between \( v_{img} \) and \( v_{txt} \).
4. Return the results with the highest similarity scores.

## 3.4. Example

- **Input**: "A photo of a dog"
- **Output**: CLIP retrieves images of dogs from the dataset.

---

# 4. FAISS

## 4.1. What is FAISS?

**FAISS** (Facebook AI Similarity Search) is an open-source library by Facebook for **approximate nearest neighbor search**, optimized for large-scale vector searches.

![alt text](/image/hcmc_ai_competition_2025/{71DF76D2-59F8-4A3A-BD9B-7593D898441C}.png)

## 4.2. Why Use FAISS?

- Linear search is slow for millions of image or text vectors.
- FAISS enables fast retrieval of top-K nearest vectors with high efficiency.

## 4.3. How FAISS Works

1. Encode all images and texts into vectors using CLIP.
2. Index the vectors using FAISS (e.g., with IVF or HNSW structures).
3. For a query, encode it and use FAISS to find the top-K nearest vectors in the index.

## 4.4. Basic Example in Python

```python
import faiss
import numpy as np

# Assume 1000 image vectors (512-dimensional)
image_vectors = np.random.rand(1000, 512).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(512)  # L2 distance
index.add(image_vectors)

# Query
query = np.random.rand(1, 512).astype('float32')
distances, indices = index.search(query, k=5)  # Find top-5 nearest images
```

---

*Last updated: July 30, 2025*
```