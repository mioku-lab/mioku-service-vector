---
title: 向量存储服务配置
description: 选择本地向量模型与下载源，配置改动需重启生效
fields:
  - key: base.enabled
    label: 启用向量化
    type: switch
    description: 关闭后只做 FTS5 关键词检索，不加载模型、不占用内存。
    default: true

  - key: base.model
    label: 本地向量模型
    type: select
    description: 全部在本机推理。
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
    description: 留空使用 HuggingFace 官方地址。
    placeholder: https://hf-mirror.com

  - key: base.cacheEnabled
    label: 缓存检索结果
    type: switch
    description: 同一句查询在有效期内直接复用上次结果。写入记忆后缓存会按库整体失效，不会读到旧数据
    default: true

  - key: base.cacheTtlMs
    label: 缓存有效期（毫秒）
    type: number
    description: 0 表示不缓存。聊天插件会在一轮对话里复用同一检索，默认 120000（2 分钟）
    default: 120000

  - key: base.cacheMaxEntries
    label: 缓存条数上限
    type: number
    description: 超出后按最久未使用淘汰，默认 256
    default: 256
---
