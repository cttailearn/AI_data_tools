
# 🤖 AI文本处理工具

基于Python 开发的强大文本处理工具，支持多种文件格式和AI模型，提供批量处理、自定义任务和实时进度监控功能。

## 🚀 主要功能

- **多模型支持**：DeepSeek、OpenAI、Claude等主流AI模型
- **文件格式支持**：TXT、CSV、Excel、PDF、Word、Markdown
- **智能处理**：摘要生成、关键词提取、情感分析、翻译等任务
- **批量处理**：多线程并发处理，实时进度监控
- **自定义任务**：创建和管理自己的文本处理任务
- **结果导出**：自动保存处理结果到指定目录

## ⚙️ 安装指南

### 前提条件
- Python 3.12
- 支持的AI模型API密钥（如OpenAI、DeepSeek等）

### 安装步骤
1. 克隆仓库：
   ```bash
   git clone [https://github.com/yourusername/ai-text-processor.git](https://github.com/cttailearn/AI_data_tools.git)
   cd AI_data_tools
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 配置API密钥：
   在`model_configs.json`文件中添加您的API密钥：
   ```json
   {
     "OpenAI GPT-4": {
       "base_url": "https://api.openai.com/v1",
       "model_name": "gpt-4",
       "api_key": "your-openai-api-key"
     },
     "DeepSeek": {
       "base_url": "https://api.deepseek.com/v1",
       "model_name": "deepseek-chat",
       "api_key": "your-deepseek-api-key"
     }
   }
   ```

## 🖥️ 使用说明

### 启动应用
```bash
python ai_text_webui.py
```

应用将在本地启动，访问 `http://localhost:7863` 使用

### 界面导航
1. **模型配置**：
   - 选择预设模型或添加自定义模型
   - 测试模型连接状态

2. **任务管理**：
   - 使用内置任务（摘要、翻译等）
   - 创建和管理自定义任务

3. **文件处理**：
   - 上传文件（TXT/CSV/Excel/PDF等）
   - 选择处理列和任务
   - 设置保存位置
   - 启动处理并查看实时进度

### 功能演示
1. 加载AI模型
2. 上传文本文件或表格文件
3. 选择处理任务（如"文本摘要"）
4. 设置并发数（建议3-5线程）
5. 点击"开始处理"查看实时日志
6. 处理完成后下载结果文件

## 📂 文件支持

| 格式       | 支持功能               | 说明                     |
|------------|-----------------------|--------------------------|
| TXT        | 全文处理              | UTF-8/GBK自动检测        |
| CSV        | 列选择处理            | 自动检测分隔符           |
| Excel      | 多工作表支持          | 处理首个工作表           |
| PDF        | 文本提取              | 需要PyPDF2库             |
| Word       | 段落提取              | 需要python-docx库        |
| Markdown   | 段落提取              | 保留原始结构             |

## ⚠️ 注意事项

1. **API限制**：
   - 确保API密钥有足够配额
   - 长文本可能被截断（>10k字符）
   - 注意模型费率（如GPT-4成本较高）

2. **处理建议**：
   - 大型文件分批处理
   - 首次使用先测试小样本
   - 监控处理日志中的错误信息

3. **依赖说明**：
   - PDF处理需要`PyPDF2`
   - Word处理需要`python-docx`
   - 编码检测需要`chardet`

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源
```
