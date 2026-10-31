---
title: 向量存储服务配置
description: 选择本地向量模型与下载源。模型在服务启动后自动准备，就绪前检索自动退化为关键词；配置改动需重启生效
fields:
  - key: base.enabled
    label: 启用向量化
    type: switch
    description: 关闭后只做 FTS5 关键词检索，不加载模型、不占用内存。适合只想要关键词记忆的场景
    default: true

  - key: base.model
    label: 本地向量模型
    type: select
    description: 全部在本机推理。首次使用会在下次启动时自动下载
    options:
      - value: bge-small-zh
        label: bge-small-zh-v1.5（24MB，推荐）
      - value: bge-base-zh
        label: bge-base-zh-v1.5（103MB，更准）
      - value: multilingual-mini
        label: paraphrase-multilingual-MiniLM（118MB，多语言）
    default: bge-small-zh

  - key: base.autoDownload
    label: 启动时自动下载
    type: switch
    description: 打开时服务启动后自动准备模型；关闭后需手动触发
    default: true

  - key: base.hfMirror
    label: 模型下载源
    type: text
    description: 留空使用 HuggingFace 官方地址。国内网络建议填镜像，例如 https://hf-mirror.com
    placeholder: https://hf-mirror.com
---
