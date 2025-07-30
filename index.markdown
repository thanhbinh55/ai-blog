---
layout: default
title: Trang chủ
---

# 🧠 Blog học AI của tôi

Chào mừng đến với blog! Đây là những bài viết gần đây:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>({{ post.date | date: "%Y-%m-%d" }})</small>
    </li>
  {% endfor %}
</ul>
