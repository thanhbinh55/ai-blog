---
layout: post
title: "AI Challenge 2025 – Track 3: Text-Image Retrieval Explained"
date: 2025-07-30
categories: [competition, ai, retrieval]
tags: [clip, faiss, jekyll, ai2025]
---

🎯 **Track 3 của HCMC AI Competition 2025** tập trung vào một chủ đề hấp dẫn: **Text-Image Retrieval**, nơi AI phải "hiểu" văn bản và tìm ra hình ảnh phù hợp nhất.

## 🔍 Retrieval là gì?

Text-Image Retrieval gồm 2 hướng chính:
- **Text → Image**: nhập mô tả, trả về hình ảnh khớp nhất.
- **Image → Text**: nhập hình ảnh, tìm văn bản mô tả tương ứng.

## 📊 Cách đánh giá mô hình

- `Recall@K`: ảnh đúng nằm trong top-K kết quả.
- `mAP`: đánh giá chất lượng và thứ tự kết quả.
- `Mean Rank`: vị trí trung bình của ảnh đúng.

Ví dụ:
```python
def recall_at_k(rank_list, k):
    return sum([1 if rank < k else 0 for rank in rank_list]) / len(rank_list)
