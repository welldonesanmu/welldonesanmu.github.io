---
title: "Git"
date: 2023-04-10T14:36:09+08:00
lastmod: 2023-04-10T14:36:09+08:00 #更新时间
draft: false
author: ["Sanmu"] #作者
comments: true #是否展示评论
tags:
  - Git         
---

# Git 推送的用法及常见指令

echo "# Blog" >> README.md \
git init \
git add README.md \
git commit -m "first commit" \
git branch -M main \
git remote add origin git@github.com:welldonesanmu/Blog.git \
git push -u origin main 



**.md 文件换行：在每一行的末尾加上 "\" ,然后再回车键。**\
新增了一个脚本，用于每次添加完之后自动deploy.