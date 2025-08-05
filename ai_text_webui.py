import gradio as gr
import os
import json
import pandas as pd
import openai
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import re
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# 可选依赖
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# 配置日志 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('ai_text_processing.log', encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

# ==================== AI模型配置 ====================

@dataclass
class ModelConfig:
    """AI模型配置类"""
    name: str
    base_url: str
    api_key: str
    model_name: str
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """配置验证"""
        if not self.base_url or not self.api_key or not self.model_name:
            raise ValueError("base_url, api_key, model_name 不能为空")
        
        # 验证URL格式
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("base_url 格式无效")

class AIModelClient:
    """AI模型客户端"""
    @classmethod
    def load_custom_configs(cls) -> Dict[str, Dict]:
        """加载自定义模型配置"""
        config_file = "model_configs.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载模型配置失败: {str(e)}")
        return {}
    
    @classmethod
    def save_custom_config(cls, name: str, config: Dict[str, Any]):
        """保存自定义模型配置"""
        config_file = "model_configs.json"
        try:
            configs = cls.load_custom_configs()
            configs[name] = config
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(configs, f, ensure_ascii=False, indent=2)
            logger.info(f"模型配置 {name} 已保存")
        except Exception as e:
            logger.error(f"保存模型配置失败: {str(e)}")
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict]:
        """获取所有模型配置（预设+自定义）"""
        # 预设模型配置
        all_configs = {
            "OpenAI GPT-4": {
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-4",
                "api_key": "your-openai-api-key"
            },
            "OpenAI GPT-3.5": {
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-3.5-turbo",
                "api_key": "your-openai-api-key"
            },
            "Claude-3": {
                "base_url": "https://api.anthropic.com/v1",
                "model_name": "claude-3-sonnet-20240229",
                "api_key": "your-anthropic-api-key"
            }
        }
        
        # 加载自定义配置并合并
        custom_configs = cls.load_custom_configs()
        all_configs.update(custom_configs)
        return all_configs
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        logger.info(f"AI模型客户端已初始化: {config.name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    def process_text(self, text: str, prompt: str) -> str:
        """使用AI模型处理文本"""
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的文本处理助手，请根据用户的要求处理文本内容。"},
                {"role": "user", "content": f"请根据以下要求处理文本：\n\n要求：{prompt}\n\n文本内容：{text}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI处理失败: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.process_text("测试", "请回复'连接成功'")
            return "连接成功" in response or "成功" in response
        except Exception:
            return False

# ==================== 文件处理模块 ====================

class FileProcessor:
    """文件处理器"""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """检测文件编码"""
        # 如果有chardet库，优先使用
        if HAS_CHARDET:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # 读取前10KB用于检测
                result = chardet.detect(raw_data)
                if result['encoding'] and result['confidence'] > 0.7:
                    return result['encoding']
            except Exception:
                pass
        
        # 备用方案：尝试常见编码
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return 'utf-8'  # 默认返回utf-8
    
    @staticmethod
    def read_txt_file(file_path: str) -> pd.DataFrame:
        """读取TXT文件"""
        try:
            encoding = FileProcessor.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 按行分割，创建DataFrame
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({'content': lines})
            return df
            
        except Exception as e:
            raise Exception(f"读取TXT文件失败: {str(e)}")
    
    @staticmethod
    def read_md_file(file_path: str) -> pd.DataFrame:
        """读取Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按段落分割（双换行符）
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            df = pd.DataFrame({'content': paragraphs})
            return df
            
        except Exception as e:
            raise Exception(f"读取Markdown文件失败: {str(e)}")
    
    @staticmethod
    def read_excel_file(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """读取Excel文件，返回数据和工作表名称"""
        try:
            # 获取所有工作表名称
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # 读取第一个工作表
            df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            
            return df, sheet_names
            
        except Exception as e:
            raise Exception(f"读取Excel文件失败: {str(e)}")
    
    @staticmethod
    def read_csv_file(file_path: str) -> pd.DataFrame:
        """读取CSV文件"""
        try:
            encoding = FileProcessor.detect_encoding(file_path)
            
            # 尝试不同的分隔符
            separators = [',', ';', '\t', '|']
            
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    # 检查是否成功解析（至少有2列或多行）
                    if len(df.columns) > 1 or len(df) > 1:
                        return df
                except Exception:
                    continue
            
            # 如果都失败，使用默认逗号分隔符
            df = pd.read_csv(file_path, encoding=encoding)
            return df
            
        except Exception as e:
            raise Exception(f"读取CSV文件失败: {str(e)}")
    
    @staticmethod
    def read_pdf_file(file_path: str) -> pd.DataFrame:
        """读取PDF文件"""
        if not HAS_PDF:
            raise Exception("需要安装PyPDF2库来读取PDF文件: pip install PyPDF2")
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text.strip())
                
                df = pd.DataFrame({'content': text_content})
                return df
                
        except Exception as e:
            raise Exception(f"读取PDF文件失败: {str(e)}")
    
    @staticmethod
    def read_docx_file(file_path: str) -> pd.DataFrame:
        """读取Word文档"""
        if not HAS_DOCX:
            raise Exception("需要安装python-docx库来读取Word文档: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            paragraphs = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    paragraphs.append(text)
            
            df = pd.DataFrame({'content': paragraphs})
            return df
            
        except Exception as e:
            raise Exception(f"读取Word文档失败: {str(e)}")
    
    @staticmethod
    def save_file(df: pd.DataFrame, original_path: str, suffix: str = "_processed") -> str:
        """保存处理后的文件到原文件目录"""
        try:
            original_path = Path(original_path)
            output_dir = original_path.parent
            
            # 生成输出文件名
            base_name = original_path.stem
            extension = original_path.suffix.lower()
            
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{base_name}{suffix}{extension}"
            
            # 根据文件类型保存
            if extension == '.csv':
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
            elif extension in ['.xlsx', '.xls']:
                df.to_excel(output_file, index=False)
            else:  # txt, md等文本文件
                # 如果只有一列，直接保存内容
                if len(df.columns) == 1:
                    content = '\n'.join(df.iloc[:, 0].astype(str))
                else:
                    # 多列时保存为CSV格式
                    content = df.to_csv(index=False, sep='\t')
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return str(output_file)
            
        except Exception as e:
            raise Exception(f"保存文件失败: {str(e)}")

# ==================== 提示词任务管理 ====================

class PromptTaskManager:
    """提示词任务管理器"""
    
    def __init__(self):
        self.tasks_file = "prompt_tasks.json"
        self.default_tasks = {
            "文本摘要": "请为以下文本生成一个简洁的摘要，突出主要内容和关键信息。",
            "关键词提取": "请从以下文本中提取5-10个关键词，用逗号分隔。",
            "情感分析": "请分析以下文本的情感倾向，回答积极、消极或中性，并简要说明理由。",
            "文本分类": "请对以下文本进行分类，并说明分类理由。",
            "问题生成": "请根据以下文本内容生成3-5个相关问题。",
            "文本翻译": "请将以下文本翻译成英文。",
            "内容扩展": "请基于以下文本内容进行扩展，增加更多细节和信息。",
            "格式化": "请将以下文本进行格式化整理，使其更加规范和易读。"
        }
        self.tasks = self.load_tasks()
    
    def load_tasks(self) -> Dict[str, str]:
        """加载提示词任务"""
        tasks = self.default_tasks.copy()
        
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    saved_tasks = json.load(f)
                # 合并默认任务和保存的任务
                tasks.update(saved_tasks)
                logger.info(f"已加载 {len(saved_tasks)} 个自定义任务")
        except Exception as e:
            logger.warning(f"加载任务文件失败: {str(e)}")
        
        return tasks
    
    def save_tasks(self):
        """保存提示词任务（保存所有被修改的任务）"""
        try:
            # 保存所有任务（包括被修改的默认任务）
            custom_tasks = {k: v for k, v in self.tasks.items() if k not in self.default_tasks or v != self.default_tasks.get(k)}
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(custom_tasks, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(custom_tasks)} 个任务（包括修改的默认任务）")
        except Exception as e:
            logger.error(f"保存任务文件失败: {str(e)}")
    
    def add_task(self, name: str, prompt: str):
        """添加或更新任务"""
        self.tasks[name] = prompt
        self.save_tasks()
        logger.info(f"任务 '{name}' 已保存")
    
    def delete_task(self, name: str) -> bool:
        """删除任务（默认任务会重置为原始值）"""
        if name in self.default_tasks:
            # 如果是默认任务，重置为原始值
            self.tasks[name] = self.default_tasks[name]
            self.save_tasks()
            logger.info(f"默认任务 '{name}' 已重置为原始值")
            return True
        
        if name in self.tasks:
            del self.tasks[name]
            self.save_tasks()
            logger.info(f"自定义任务 '{name}' 已删除")
            return True
        return False
    
    def get_task_names(self) -> List[str]:
        """获取所有任务名称"""
        return list(self.tasks.keys())
    
    def get_task_prompt(self, name: str) -> str:
        """获取任务提示词"""
        return self.tasks.get(name, "")
    
    def is_default_task(self, name: str) -> bool:
        """判断是否为默认任务"""
        return name in self.default_tasks
    
    def reload_tasks(self):
        """重新加载任务"""
        self.tasks = self.load_tasks()

# ==================== 全局变量 ====================

# 全局变量
current_model_client = None
current_dataframe = None
original_file_path = None
task_manager = PromptTaskManager()

# ==================== Gradio界面函数 ====================

def load_model(preset_name: str, custom_name: str, custom_base_url: str, 
               custom_api_key: str, custom_model_name: str, save_custom: bool = False) -> Tuple[str, str]:
    """加载AI模型"""
    global current_model_client
    
    try:
        if preset_name != "自定义":
            # 使用预设或已保存的配置
            all_configs = AIModelClient.get_all_configs()
            if preset_name not in all_configs:
                return f"错误：未知的模型配置 {preset_name}", ""
            
            config_dict = all_configs[preset_name].copy()
            config = ModelConfig(**config_dict)
        else:
            # 使用自定义配置
            if not all([custom_name, custom_base_url, custom_api_key, custom_model_name]):
                return "错误：自定义配置信息不完整", ""
            
            config = ModelConfig(
                name=custom_name,
                base_url=custom_base_url,
                api_key=custom_api_key,
                model_name=custom_model_name
            )
            
            # 保存自定义配置
            if save_custom and custom_name:
                config_dict = {
                    "name": custom_name,
                    "base_url": custom_base_url,
                    "api_key": custom_api_key,
                    "model_name": custom_model_name
                }
                AIModelClient.save_custom_config(custom_name, config_dict)
        
        # 创建客户端
        current_model_client = AIModelClient(config)
        
        # 测试连接
        logger.info(f"正在测试模型连接: {config.name}")
        if current_model_client.test_connection():
            message = f"✅ 模型 {config.name} 加载成功并连接正常"
            logger.info(message)
            # 更新配置选择列表
            updated_choices = list(AIModelClient.get_all_configs().keys()) + ["自定义"]
            return message, gr.Dropdown(choices=updated_choices)
        else:
            message = f"⚠️ 模型 {config.name} 加载成功但连接测试失败，请检查配置"
            logger.warning(message)
            updated_choices = list(AIModelClient.get_all_configs().keys()) + ["自定义"]
            return message, gr.Dropdown(choices=updated_choices)
            
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""

def handle_file_upload(file) -> Tuple[str, str, gr.Dropdown]:
    """上传并读取文件"""
    global current_dataframe, original_file_path
    
    if file is None:
        return "请选择文件", "", gr.Dropdown(choices=[], visible=False)
    
    try:
        file_path = file.name
        original_file_path = file_path
        file_extension = Path(file_path).suffix.lower()
        
        # 根据文件类型读取
        if file_extension == '.txt':
            df = FileProcessor.read_txt_file(file_path)
        elif file_extension == '.md':
            df = FileProcessor.read_md_file(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df, sheet_names = FileProcessor.read_excel_file(file_path)
        elif file_extension == '.csv':
            df = FileProcessor.read_csv_file(file_path)
        elif file_extension == '.pdf':
            df = FileProcessor.read_pdf_file(file_path)
        elif file_extension in ['.docx', '.doc']:
            df = FileProcessor.read_docx_file(file_path)
        else:
            return f"不支持的文件格式: {file_extension}\n支持的格式: .txt, .md, .csv, .xlsx, .xls, .pdf, .docx", "", gr.Dropdown(choices=[], visible=False)
        
        current_dataframe = df
        
        # 生成列选择选项（支持多选）
        columns = df.columns.tolist()
        
        # 对于表格文件（Excel/CSV），显示列选择；对于文本文件，隐藏列选择
        if file_extension in ['.xlsx', '.xls', '.csv']:
            column_dropdown = gr.Dropdown(
                choices=columns, 
                value=None,
                label="选择要处理的列",
                multiselect=True,
                visible=True
            )
        else:
            # 对于文本文件，默认选择第一列
            column_dropdown = gr.Dropdown(
                choices=columns,
                value=[columns[0]] if columns else None,
                label="选择要处理的列",
                multiselect=True,
                visible=False
            )
        
        # 生成预览
        preview = df.head(10).to_string(index=False, max_cols=5, max_colwidth=50)
        
        # 添加文件信息
        file_info = f"文件类型: {file_extension}\n文件大小: {os.path.getsize(file_path) / 1024:.1f} KB\n数据行数: {len(df)}\n数据列数: {len(df.columns)}"
        
        return f"✅ 文件读取成功\n{file_info}", preview, column_dropdown
        
    except Exception as e:
        return f"❌ 文件读取失败: {str(e)}", "", gr.Dropdown(choices=[], visible=False)

def get_task_prompt(task_name: str) -> str:
    """获取任务提示词"""
    return task_manager.get_task_prompt(task_name)

def add_custom_task(task_name: str, task_prompt: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """添加自定义任务"""
    if not task_name or not task_prompt:
        task_choices = task_manager.get_task_names()
        return "任务名称和提示词不能为空", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    try:
        task_manager.add_task(task_name, task_prompt)
        task_choices = task_manager.get_task_names()
        updated_dropdown1 = gr.Dropdown(choices=task_choices, value=task_name)
        updated_dropdown2 = gr.Dropdown(choices=task_choices, value=task_name)
        return f"✅ 任务 '{task_name}' 添加成功", updated_dropdown1, updated_dropdown2
    except Exception as e:
        task_choices = task_manager.get_task_names()
        return f"❌ 添加任务失败: {str(e)}", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)

def delete_task(task_name: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """删除任务（默认任务会重置为原始值）"""
    if not task_name:
        task_choices = task_manager.get_task_names()
        return "请选择要删除的任务", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    try:
        is_default = task_manager.is_default_task(task_name)
        if task_manager.delete_task(task_name):
            task_choices = task_manager.get_task_names()
            updated_dropdown1 = gr.Dropdown(choices=task_choices)
            updated_dropdown2 = gr.Dropdown(choices=task_choices)
            if is_default:
                return f"✅ 默认任务 '{task_name}' 已重置为原始值", updated_dropdown1, updated_dropdown2
            else:
                return f"✅ 自定义任务 '{task_name}' 删除成功", updated_dropdown1, updated_dropdown2
        else:
            task_choices = task_manager.get_task_names()
            return f"❌ 无法删除任务 '{task_name}'", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    except Exception as e:
        task_choices = task_manager.get_task_names()
        return f"❌ 删除任务失败: {str(e)}", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)

def edit_task(task_name: str, new_prompt: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """编辑任务提示词"""
    if not task_name:
        task_choices = task_manager.get_task_names()
        return "请选择要编辑的任务", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    if not new_prompt.strip():
        task_choices = task_manager.get_task_names()
        return "提示词不能为空", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    try:
        task_manager.add_task(task_name, new_prompt.strip())
        task_choices = task_manager.get_task_names()
        updated_dropdown1 = gr.Dropdown(choices=task_choices, value=task_name)
        updated_dropdown2 = gr.Dropdown(choices=task_choices, value=task_name)
        is_default = task_manager.is_default_task(task_name)
        if is_default:
            return f"✅ 默认任务 '{task_name}' 修改成功", updated_dropdown1, updated_dropdown2
        else:
            return f"✅ 任务 '{task_name}' 修改成功", updated_dropdown1, updated_dropdown2
    except Exception as e:
        task_choices = task_manager.get_task_names()
        return f"❌ 编辑任务失败: {str(e)}", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)

def reload_tasks() -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """重新加载任务"""
    try:
        task_manager.reload_tasks()
        task_choices = task_manager.get_task_names()
        updated_dropdown1 = gr.Dropdown(choices=task_choices)
        updated_dropdown2 = gr.Dropdown(choices=task_choices)
        return "✅ 任务列表已重新加载", updated_dropdown1, updated_dropdown2
    except Exception as e:
        task_choices = task_manager.get_task_names()
        return f"❌ 重新加载失败: {str(e)}", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)

def process_data_stream(file_upload, selected_columns, task_name: str, 
                        batch_size: int = 10, max_workers: int = 3, 
                        save_location: str = "当前文件的output目录", custom_save_path: str = ""):
    """流式处理数据（支持多线程和多列选择，实时进度显示）"""
    global current_model_client, current_dataframe, original_file_path
    
    if current_model_client is None:
        yield "❌ 请先加载AI模型", "", 0.0
        return
    
    if current_dataframe is None:
        yield "❌ 请先上传文件", "", 0.0
        return
    
    if not selected_columns:
        yield "❌ 请选择要处理的列", "", 0.0
        return
    
    if not task_name:
        yield "❌ 请选择处理任务", "", 0.0
        return
    
    try:
        # 获取任务提示词
        prompt = task_manager.get_task_prompt(task_name)
        if not prompt:
            yield "❌ 选择的任务无效", "", 0.0
            return
        
        # 处理多列选择
        if isinstance(selected_columns, str):
            columns_to_process = [selected_columns]
        else:
            columns_to_process = selected_columns if selected_columns else []
        
        if not columns_to_process:
            yield "❌ 请选择要处理的列", "", 0.0
            return
        
        # 验证列是否存在
        missing_columns = [col for col in columns_to_process if col not in current_dataframe.columns]
        if missing_columns:
            yield f"❌ 以下列不存在: {', '.join(missing_columns)}", "", 0.0
            return
        
        total_processed = 0
        processing_log = []
        overall_progress = 0.0
        total_start_time = time.time()  # 记录总开始时间
        
        # 计算总的处理项目数
        total_items_all_columns = 0
        for column in columns_to_process:
            data_to_process = current_dataframe[column].astype(str).tolist()
            indexed_data = [(i, item) for i, item in enumerate(data_to_process) if item.strip()]
            total_items_all_columns += len(indexed_data)
        
        # 初始化信息
        processing_log.append(f"📊 开始处理 {len(columns_to_process)} 列，共 {total_items_all_columns} 条数据")
        processing_log.append(f"⚙️ 使用 {max_workers} 个线程并发处理")
        processing_log.append(f"📁 默认输出目录: {Path(__file__).parent / 'output'}")
        processing_log.append(f"📂 保存位置设置: {save_location}")
        if save_location == "自定义目录" and custom_save_path.strip():
            processing_log.append(f"📍 自定义路径: {custom_save_path.strip()}")
        processing_log.append(f"📝 处理任务: {task_name}")
        processing_log.append("" + "="*50)
        
        # 输出初始状态
        yield "\n".join(processing_log), "", 0.0
        
        global_processed_count = 0
        
        # 处理每一列
        for col_index, column in enumerate(columns_to_process):
            processing_log.append(f"\n🔄 正在处理列: {column} ({col_index + 1}/{len(columns_to_process)})")
            processing_log.append(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取要处理的数据
            data_to_process = current_dataframe[column].astype(str).tolist()
            
            # 过滤空值并保存原始索引
            indexed_data = [(i, item) for i, item in enumerate(data_to_process) if item.strip()]
            
            if not indexed_data:
                processing_log.append(f"⚠️ 列 {column} 中没有有效数据，跳过")
                yield "\n".join(processing_log), "", overall_progress
                continue
            
            total_items = len(indexed_data)
            processing_log.append(f"📝 该列有效数据: {total_items} 条")
            processing_log.append(f"📊 数据长度统计: 平均 {sum(len(str(item[1])) for item in indexed_data) / len(indexed_data):.0f} 字符")
            processing_log.append(f"🚀 开始并发处理，线程数: {max_workers}")
            processing_log.append("-" * 40)
            
            # 输出列开始处理状态
            yield "\n".join(processing_log), "", overall_progress
            
            # 创建结果字典
            results_dict = {}
            processed_count = 0
            start_time = time.time()
            
            def process_single_item(indexed_item):
                """处理单个文本项"""
                index, item = indexed_item
                try:
                    # 限制单个文本长度
                    if len(item) > 10000:
                        item = item[:10000] + "...[文本过长，已截断]"
                    
                    result = current_model_client.process_text(item, prompt)
                    return index, result, True
                except Exception as e:
                    error_msg = f"处理失败: {str(e)}"
                    logger.error(f"处理第 {index+1} 项失败: {str(e)}")
                    return index, error_msg, False
            
            # 使用线程池进行并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(process_single_item, item): item[0] for item in indexed_data}
                
                # 收集结果
                for future in as_completed(future_to_index):
                    try:
                        index, result, success = future.result()
                        results_dict[index] = result
                        processed_count += 1
                        global_processed_count += 1
                        
                        # 计算进度
                        column_progress = processed_count / total_items * 100
                        overall_progress = global_processed_count / total_items_all_columns * 100
                        
                        # 计算处理速度
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            speed = processed_count / elapsed_time
                            remaining_items = total_items - processed_count
                            eta = remaining_items / speed if speed > 0 else 0
                            
                            # 实时详细的进度显示（每处理一个就更新）
                            if processed_count % 1 == 0 or processed_count == total_items:
                                current_log = processing_log.copy()
                                status_msg = (
                                    f"🔄 实时状态 | 列: {column} ({col_index + 1}/{len(columns_to_process)}) | "
                                    f"当前列进度: {processed_count}/{total_items} ({column_progress:.1f}%) | "
                                    f"总体进度: {global_processed_count}/{total_items_all_columns} ({overall_progress:.1f}%) | "
                                    f"处理速度: {speed:.1f}条/秒 | 预计剩余时间: {eta:.0f}秒"
                                )
                                current_log.append(status_msg)
                                
                                # 显示最近处理的内容预览（成功的情况）
                                if success and len(result) > 0:
                                    preview_text = result[:50] + "..." if len(result) > 50 else result
                                    current_log.append(f"   ✅ 最新处理结果预览: {preview_text}")
                                elif not success:
                                    current_log.append(f"   ❌ 处理失败: {result}")
                                
                                # 实时输出状态
                                yield "\n".join(current_log), "", overall_progress
                        
                    except Exception as e:
                        logger.error(f"获取处理结果失败: {str(e)}")
                        global_processed_count += 1
            
            # 构建完整的结果列表
            full_results = []
            for i, original_item in enumerate(data_to_process):
                if i in results_dict:
                    full_results.append(results_dict[i])
                else:
                    full_results.append("" if not original_item.strip() else "处理失败")
            
            # 添加结果列到DataFrame
            result_column_name = f"{column}_processed"
            current_dataframe[result_column_name] = full_results
            
            successful_count = len([r for r in full_results if r and not r.startswith('处理失败')])
            failed_count = total_items - successful_count
            total_processed += successful_count
            
            # 计算该列的处理时间
            column_end_time = time.time()
            column_duration = column_end_time - start_time
            
            processing_log.append("-" * 40)
            processing_log.append(f"✅ 列 {column} 处理完成统计:")
            processing_log.append(f"   📊 成功处理: {successful_count} 条 ({successful_count/total_items*100:.1f}%)")
            processing_log.append(f"   ❌ 处理失败: {failed_count} 条 ({failed_count/total_items*100:.1f}%)")
            processing_log.append(f"   ⏱️ 处理耗时: {column_duration:.1f} 秒")
            processing_log.append(f"   🚀 平均速度: {total_items/column_duration:.1f} 条/秒")
            processing_log.append(f"   ⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            processing_log.append(f"   📈 累计完成: {total_processed}/{total_items_all_columns} 条")
            
            # 输出列完成状态
            yield "\n".join(processing_log), "", overall_progress
        
        # 保存结果
        processing_log.append("\n" + "="*50)
        processing_log.append("💾 正在保存处理结果...")
        processing_log.append(f"📁 输出目录: {Path(__file__).parent / 'output'}")
        
        # 输出保存开始状态
        yield "\n".join(processing_log), "", overall_progress
        
        if original_file_path:
            try:
                if save_location == "自定义目录" and custom_save_path.strip():
                    # 使用自定义保存路径
                    custom_dir = Path(custom_save_path.strip())
                    custom_dir.mkdir(parents=True, exist_ok=True)
                    
                    original_name = Path(original_file_path).stem
                    original_ext = Path(original_file_path).suffix
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{original_name}_processed_{timestamp}{original_ext}"
                    output_path = custom_dir / output_filename
                    
                    processing_log.append(f"📂 使用自定义目录: {custom_dir}")
                    
                    if original_ext.lower() in ['.xlsx', '.xls']:
                        current_dataframe.to_excel(str(output_path), index=False)
                        processing_log.append(f"📊 Excel文件保存中...")
                    else:
                        current_dataframe.to_csv(str(output_path), index=False, encoding='utf-8-sig')
                        processing_log.append(f"📄 CSV文件保存中...")
                else:
                    # 使用默认output目录（当前代码文件所在目录下的output文件夹）
                    processing_log.append(f"📂 使用默认输出目录: {Path(__file__).parent / 'output'}")
                    output_path = save_to_output_dir(current_dataframe, original_file_path)
                    processing_log.append(f"💾 文件保存中...")
                
                processing_log.append(f"✅ 结果已成功保存到: {output_path}")
                processing_log.append(f"📊 保存的数据行数: {len(current_dataframe)}")
                processing_log.append(f"📋 保存的数据列数: {len(current_dataframe.columns)}")
                
                # 生成结果预览
                result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
                
                # 计算总处理时间
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                
                # 生成最终统计信息
                processing_log.append("\n" + "="*50)
                processing_log.append("🎯 === 最终处理统计报告 === 🎯")
                processing_log.append(f"📋 处理任务: {task_name}")
                processing_log.append(f"📊 处理列数: {len(columns_to_process)} 列")
                processing_log.append(f"📈 总数据量: {total_items_all_columns} 条")
                processing_log.append(f"✅ 成功处理: {total_processed} 条")
                processing_log.append(f"❌ 失败数量: {total_items_all_columns - total_processed} 条")
                processing_log.append(f"📊 总成功率: {(total_processed/total_items_all_columns*100):.1f}%")
                processing_log.append(f"🔧 并发线程: {max_workers} 个")
                processing_log.append(f"⏱️ 总处理时间: {total_duration:.1f} 秒")
                processing_log.append(f"🚀 平均处理速度: {total_items_all_columns/total_duration:.1f} 条/秒")
                processing_log.append(f"💾 输出位置: {output_path}")
                processing_log.append(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                processing_log.append("="*50)
                processing_log.append(f"🎉 所有处理任务已成功完成！")
                
                final_message = "\n".join(processing_log)
                
                # 最终输出
                yield final_message, result_preview, 100.0
                
            except Exception as e:
                logger.error(f"保存文件失败: {str(e)}")
                processing_log.append(f"❌ 保存失败: {str(e)}")
                
                result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
                final_message = "\n".join(processing_log)
                final_message += f"\n\n⚠️ 处理完成但保存失败，共处理 {total_processed} 条数据"
                
                yield final_message, result_preview, overall_progress
        else:
            processing_log.append(f"🎉 总计处理完成 {total_processed} 条数据")
            result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
            yield "\n".join(processing_log), result_preview, overall_progress
            
    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        yield f"❌ 处理失败: {str(e)}", "", 0.0

def process_data(file_upload, selected_columns, task_name: str, 
                batch_size: int = 10, max_workers: int = 3, 
                save_location: str = "当前文件的output目录", custom_save_path: str = "") -> Tuple[str, str, float]:
    """处理数据（支持多线程和多列选择，带进度显示）"""
    global current_model_client, current_dataframe, original_file_path
    
    if current_model_client is None:
        return "❌ 请先加载AI模型", "", 0.0
    
    if current_dataframe is None:
        return "❌ 请先上传文件", "", 0.0
    
    if not selected_columns:
        return "❌ 请选择要处理的列", "", 0.0
    
    if not task_name:
        return "❌ 请选择处理任务", "", 0.0
    
    try:
        # 获取任务提示词
        prompt = task_manager.get_task_prompt(task_name)
        if not prompt:
            return "❌ 选择的任务无效", "", 0.0
        
        # 处理多列选择
        if isinstance(selected_columns, str):
            columns_to_process = [selected_columns]
        else:
            columns_to_process = selected_columns if selected_columns else []
        
        if not columns_to_process:
            return "❌ 请选择要处理的列", "", 0.0
        
        # 验证列是否存在
        missing_columns = [col for col in columns_to_process if col not in current_dataframe.columns]
        if missing_columns:
            return f"❌ 以下列不存在: {', '.join(missing_columns)}", "", 0.0
        
        total_processed = 0
        processing_log = []
        overall_progress = 0.0
        total_start_time = time.time()  # 记录总开始时间
        
        # 计算总的处理项目数
        total_items_all_columns = 0
        for column in columns_to_process:
            data_to_process = current_dataframe[column].astype(str).tolist()
            indexed_data = [(i, item) for i, item in enumerate(data_to_process) if item.strip()]
            total_items_all_columns += len(indexed_data)
        
        processing_log.append(f"📊 开始处理 {len(columns_to_process)} 列，共 {total_items_all_columns} 条数据")
        processing_log.append(f"⚙️ 使用 {max_workers} 个线程并发处理")
        processing_log.append(f"📁 默认输出目录: {Path(__file__).parent / 'output'}")
        processing_log.append(f"📂 保存位置设置: {save_location}")
        if save_location == "自定义目录" and custom_save_path.strip():
            processing_log.append(f"📍 自定义路径: {custom_save_path.strip()}")
        processing_log.append(f"📝 处理任务: {task_name}")
        processing_log.append("" + "="*50)
        
        global_processed_count = 0
        
        # 处理每一列
        for col_index, column in enumerate(columns_to_process):
            processing_log.append(f"\n🔄 正在处理列: {column} ({col_index + 1}/{len(columns_to_process)})")
            processing_log.append(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取要处理的数据
            data_to_process = current_dataframe[column].astype(str).tolist()
            
            # 过滤空值并保存原始索引
            indexed_data = [(i, item) for i, item in enumerate(data_to_process) if item.strip()]
            
            if not indexed_data:
                processing_log.append(f"⚠️ 列 {column} 中没有有效数据，跳过")
                continue
            
            total_items = len(indexed_data)
            processing_log.append(f"📝 该列有效数据: {total_items} 条")
            processing_log.append(f"📊 数据长度统计: 平均 {sum(len(str(item[1])) for item in indexed_data) / len(indexed_data):.0f} 字符")
            processing_log.append(f"🚀 开始并发处理，线程数: {max_workers}")
            processing_log.append("-" * 40)
            
            # 创建结果字典
            results_dict = {}
            processed_count = 0
            start_time = time.time()
            
            def process_single_item(indexed_item):
                """处理单个文本项"""
                index, item = indexed_item
                try:
                    # 限制单个文本长度
                    if len(item) > 10000:
                        item = item[:10000] + "...[文本过长，已截断]"
                    
                    result = current_model_client.process_text(item, prompt)
                    return index, result, True
                except Exception as e:
                    error_msg = f"处理失败: {str(e)}"
                    logger.error(f"处理第 {index+1} 项失败: {str(e)}")
                    return index, error_msg, False
            
            # 使用线程池进行并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(process_single_item, item): item[0] for item in indexed_data}
                
                # 收集结果
                for future in as_completed(future_to_index):
                    try:
                        index, result, success = future.result()
                        results_dict[index] = result
                        processed_count += 1
                        global_processed_count += 1
                        
                        # 计算进度
                        column_progress = processed_count / total_items * 100
                        overall_progress = global_processed_count / total_items_all_columns * 100
                        
                        # 计算处理速度
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            speed = processed_count / elapsed_time
                            remaining_items = total_items - processed_count
                            eta = remaining_items / speed if speed > 0 else 0
                            
                            # 实时详细的进度显示
                            if processed_count % 2 == 0 or processed_count == total_items:
                                status_msg = (
                                    f"🔄 实时状态 | 列: {column} ({col_index + 1}/{len(columns_to_process)}) | "
                                    f"当前列进度: {processed_count}/{total_items} ({column_progress:.1f}%) | "
                                    f"总体进度: {global_processed_count}/{total_items_all_columns} ({overall_progress:.1f}%) | "
                                    f"处理速度: {speed:.1f}条/秒 | 预计剩余时间: {eta:.0f}秒"
                                )
                                processing_log.append(status_msg)
                                
                                # 显示最近处理的内容预览（成功的情况）
                                if success and len(result) > 0:
                                    preview_text = result[:50] + "..." if len(result) > 50 else result
                                    processing_log.append(f"   ✅ 最新处理结果预览: {preview_text}")
                                elif not success:
                                    processing_log.append(f"   ❌ 处理失败: {result}")
                        
                    except Exception as e:
                        logger.error(f"获取处理结果失败: {str(e)}")
                        global_processed_count += 1
            
            # 构建完整的结果列表
            full_results = []
            for i, original_item in enumerate(data_to_process):
                if i in results_dict:
                    full_results.append(results_dict[i])
                else:
                    full_results.append("" if not original_item.strip() else "处理失败")
            
            # 添加结果列到DataFrame
            result_column_name = f"{column}_processed"
            current_dataframe[result_column_name] = full_results
            
            successful_count = len([r for r in full_results if r and not r.startswith('处理失败')])
            failed_count = total_items - successful_count
            total_processed += successful_count
            
            # 计算该列的处理时间
            column_end_time = time.time()
            column_duration = column_end_time - start_time
            
            processing_log.append("-" * 40)
            processing_log.append(f"✅ 列 {column} 处理完成统计:")
            processing_log.append(f"   📊 成功处理: {successful_count} 条 ({successful_count/total_items*100:.1f}%)")
            processing_log.append(f"   ❌ 处理失败: {failed_count} 条 ({failed_count/total_items*100:.1f}%)")
            processing_log.append(f"   ⏱️ 处理耗时: {column_duration:.1f} 秒")
            processing_log.append(f"   🚀 平均速度: {total_items/column_duration:.1f} 条/秒")
            processing_log.append(f"   ⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            processing_log.append(f"   📈 累计完成: {total_processed}/{total_items_all_columns} 条")
        
        # 保存结果
        processing_log.append("\n" + "="*50)
        processing_log.append("💾 正在保存处理结果...")
        processing_log.append(f"📁 输出目录: {Path(__file__).parent / 'output'}")
        
        if original_file_path:
            try:
                if save_location == "自定义目录" and custom_save_path.strip():
                    # 使用自定义保存路径
                    custom_dir = Path(custom_save_path.strip())
                    custom_dir.mkdir(parents=True, exist_ok=True)
                    
                    original_name = Path(original_file_path).stem
                    original_ext = Path(original_file_path).suffix
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{original_name}_processed_{timestamp}{original_ext}"
                    output_path = custom_dir / output_filename
                    
                    processing_log.append(f"📂 使用自定义目录: {custom_dir}")
                    
                    if original_ext.lower() in ['.xlsx', '.xls']:
                        current_dataframe.to_excel(str(output_path), index=False)
                        processing_log.append(f"📊 Excel文件保存中...")
                    else:
                        current_dataframe.to_csv(str(output_path), index=False, encoding='utf-8-sig')
                        processing_log.append(f"📄 CSV文件保存中...")
                else:
                    # 使用默认output目录（当前代码文件所在目录下的output文件夹）
                    processing_log.append(f"📂 使用默认输出目录: {Path(__file__).parent / 'output'}")
                    output_path = save_to_output_dir(current_dataframe, original_file_path)
                    processing_log.append(f"💾 文件保存中...")
                
                processing_log.append(f"✅ 结果已成功保存到: {output_path}")
                processing_log.append(f"📊 保存的数据行数: {len(current_dataframe)}")
                processing_log.append(f"📋 保存的数据列数: {len(current_dataframe.columns)}")
                
                # 生成结果预览
                result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
                
                # 计算总处理时间
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                
                # 生成最终统计信息
                processing_log.append("\n" + "="*50)
                processing_log.append("🎯 === 最终处理统计报告 === 🎯")
                processing_log.append(f"📋 处理任务: {task_name}")
                processing_log.append(f"📊 处理列数: {len(columns_to_process)} 列")
                processing_log.append(f"📈 总数据量: {total_items_all_columns} 条")
                processing_log.append(f"✅ 成功处理: {total_processed} 条")
                processing_log.append(f"❌ 失败数量: {total_items_all_columns - total_processed} 条")
                processing_log.append(f"📊 总成功率: {(total_processed/total_items_all_columns*100):.1f}%")
                processing_log.append(f"🔧 并发线程: {max_workers} 个")
                processing_log.append(f"⏱️ 总处理时间: {total_duration:.1f} 秒")
                processing_log.append(f"🚀 平均处理速度: {total_items_all_columns/total_duration:.1f} 条/秒")
                processing_log.append(f"💾 输出位置: {output_path}")
                processing_log.append(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                processing_log.append("="*50)
                processing_log.append(f"🎉 所有处理任务已成功完成！")
                
                final_message = "\n".join(processing_log)
                
                return final_message, result_preview, 100.0
                
            except Exception as e:
                logger.error(f"保存文件失败: {str(e)}")
                processing_log.append(f"❌ 保存失败: {str(e)}")
                
                result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
                final_message = "\n".join(processing_log)
                final_message += f"\n\n⚠️ 处理完成但保存失败，共处理 {total_processed} 条数据"
                
                return final_message, result_preview, overall_progress
        else:
            processing_log.append(f"🎉 总计处理完成 {total_processed} 条数据")
            result_preview = generate_enhanced_result_preview(current_dataframe, columns_to_process, total_processed)
            return "\n".join(processing_log), result_preview, overall_progress
            
    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        return f"❌ 处理失败: {str(e)}", "", 0.0

def generate_result_preview(df: pd.DataFrame, processed_columns: List[str]) -> str:
    """生成处理结果预览"""
    try:
        preview_lines = []
        preview_lines.append("=== 处理结果预览 ===")
        preview_lines.append(f"数据总行数: {len(df)}")
        preview_lines.append(f"数据总列数: {len(df.columns)}")
        preview_lines.append("")
        
        # 显示处理的列信息
        for col in processed_columns:
            result_col = f"{col}_processed"
            if result_col in df.columns:
                non_empty_count = df[result_col].astype(str).str.strip().ne('').sum()
                preview_lines.append(f"列 '{col}' -> '{result_col}': {non_empty_count} 条有效结果")
        
        preview_lines.append("")
        preview_lines.append("=== 前5行数据预览 ===")
        
        # 显示前5行的原始数据和处理结果
        for i in range(min(5, len(df))):
            preview_lines.append(f"--- 第 {i+1} 行 ---")
            for col in processed_columns:
                original_value = str(df.iloc[i][col])[:100]
                result_col = f"{col}_processed"
                if result_col in df.columns:
                    processed_value = str(df.iloc[i][result_col])[:100]
                    preview_lines.append(f"{col}: {original_value}")
                    preview_lines.append(f"{result_col}: {processed_value}")
                    preview_lines.append("")
        
        return "\n".join(preview_lines)
        
    except Exception as e:
        return f"生成预览失败: {str(e)}"

def generate_enhanced_result_preview(df: pd.DataFrame, processed_columns: List[str], total_processed: int) -> str:
    """生成增强的处理结果预览"""
    try:
        preview_lines = []
        preview_lines.append("📊 === 处理结果详细预览 === 📊")
        preview_lines.append(f"📋 数据总行数: {len(df)}")
        preview_lines.append(f"📋 数据总列数: {len(df.columns)}")
        preview_lines.append(f"✅ 成功处理: {total_processed} 条")
        preview_lines.append("")
        
        # 显示每列的详细统计
        preview_lines.append("📈 各列处理统计:")
        for col in processed_columns:
            result_col = f"{col}_processed"
            if result_col in df.columns:
                # 统计有效结果
                valid_results = df[result_col].astype(str).str.strip()
                non_empty_count = valid_results.ne('').sum()
                failed_count = valid_results.str.startswith('处理失败').sum()
                success_count = non_empty_count - failed_count
                
                # 计算平均长度
                valid_lengths = valid_results[valid_results.ne('')].str.len()
                avg_length = valid_lengths.mean() if len(valid_lengths) > 0 else 0
                
                preview_lines.append(f"   🔹 列 '{col}':")
                preview_lines.append(f"      ✅ 成功: {success_count} 条")
                preview_lines.append(f"      ❌ 失败: {failed_count} 条")
                preview_lines.append(f"      📏 平均结果长度: {avg_length:.0f} 字符")
                preview_lines.append("")
        
        preview_lines.append("🔍 === 数据样本预览 === 🔍")
        
        # 显示前3行的原始数据和处理结果
        for i in range(min(3, len(df))):
            preview_lines.append(f"📄 第 {i+1} 行样本:")
            for col in processed_columns:
                original_value = str(df.iloc[i][col])
                result_col = f"{col}_processed"
                if result_col in df.columns:
                    processed_value = str(df.iloc[i][result_col])
                    
                    # 限制显示长度
                    original_display = original_value[:150] + "..." if len(original_value) > 150 else original_value
                    processed_display = processed_value[:150] + "..." if len(processed_value) > 150 else processed_value
                    
                    preview_lines.append(f"   📝 原始[{col}]: {original_display}")
                    preview_lines.append(f"   🤖 处理[{result_col}]: {processed_display}")
                    preview_lines.append("   " + "-"*50)
            preview_lines.append("")
        
        # 添加数据质量分析
        preview_lines.append("📊 === 数据质量分析 === 📊")
        for col in processed_columns:
            result_col = f"{col}_processed"
            if result_col in df.columns:
                results = df[result_col].astype(str)
                
                # 分析结果类型
                empty_count = results.str.strip().eq('').sum()
                error_count = results.str.contains('处理失败|错误|失败', na=False).sum()
                valid_count = len(results) - empty_count - error_count
                
                preview_lines.append(f"   📊 列 '{col}' 质量分析:")
                preview_lines.append(f"      🟢 有效结果: {valid_count} 条 ({valid_count/len(results)*100:.1f}%)")
                preview_lines.append(f"      🔴 处理错误: {error_count} 条 ({error_count/len(results)*100:.1f}%)")
                preview_lines.append(f"      ⚪ 空白结果: {empty_count} 条 ({empty_count/len(results)*100:.1f}%)")
                preview_lines.append("")
        
        return "\n".join(preview_lines)
        
    except Exception as e:
        return f"生成增强预览失败: {str(e)}"

def save_to_output_dir(df: pd.DataFrame, original_path: str, suffix: str = "_processed") -> str:
    """保存文件到当前代码文件所在的output目录"""
    try:
        # 获取当前代码文件所在目录
        current_script_dir = Path(__file__).parent
        
        # 创建output目录（在当前代码文件目录下）
        output_dir = current_script_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = Path(original_path)
        
        # 生成输出文件名
        base_name = original_path.stem
        extension = original_path.suffix.lower()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_file = output_dir / f"{base_name}{suffix}_{timestamp}{extension}"
        
        # 根据文件类型保存
        if extension == '.csv':
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif extension in ['.xlsx', '.xls']:
            df.to_excel(output_file, index=False)
        else:  # txt, md等文本文件
            # 如果只有一列，直接保存内容
            if len(df.columns) == 1:
                content = '\n'.join(df.iloc[:, 0].astype(str))
            else:
                # 多列时保存为CSV格式
                content = df.to_csv(index=False, sep='\t')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return str(output_file)
        
    except Exception as e:
        raise Exception(f"保存文件失败: {str(e)}")

def get_processed_data() -> str:
    """获取处理后的数据预览"""
    global current_dataframe
    
    if current_dataframe is None:
        return "暂无数据"
    
    try:
        # 显示前10行数据
        preview = current_dataframe.head(10).to_string(index=False, max_cols=10)
        return preview
    except Exception as e:
        return f"获取数据预览失败: {str(e)}"

# ==================== Gradio界面 ====================

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="AI文本处理工具", theme=gr.themes.Ocean()) as interface:
        gr.Markdown("# 🤖 AI文本处理工具")
        gr.Markdown("支持多种文件格式的AI智能文本处理")
        
        with gr.Tabs() as tabs:
            # 第一个标签页：模型配置
            with gr.TabItem("🔧 模型配置"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### AI模型配置")
                        preset_dropdown = gr.Dropdown(
                            choices=list(AIModelClient.get_all_configs().keys()) + ["自定义"],
                            value="deepseek",
                            label="选择模型配置"
                        )
                        
                        # 自定义配置（默认隐藏）
                        with gr.Group(visible=False) as custom_config:
                            custom_name = gr.Textbox(label="模型名称", placeholder="例如：我的模型")
                            custom_base_url = gr.Textbox(label="API地址", placeholder="例如：https://api.example.com/v1")
                            custom_api_key = gr.Textbox(label="API密钥", type="password")
                            custom_model_name = gr.Textbox(label="模型名称", placeholder="例如：gpt-3.5-turbo")
                            save_custom_config = gr.Checkbox(label="保存此配置", value=False)
                        
                        load_model_btn = gr.Button("🚀 加载模型", variant="primary")
                        model_status = gr.Textbox(label="模型状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型配置说明")
                        gr.Markdown("""
                        **预设配置：**
                        - DeepSeek: 高性能中文大模型
                        - OpenAI: GPT系列模型
                        - 本地模型: 本地部署的模型
                        
                        **自定义配置：**
                        - 支持任何兼容OpenAI API的模型
                        - 可保存配置供下次使用
                        - 自动测试连接状态
                        """)
            
            # 第二个标签页：任务管理
            with gr.TabItem("📝 任务管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 提示词任务管理")
                        task_dropdown = gr.Dropdown(
                            choices=task_manager.get_task_names(),
                            value=task_manager.get_task_names()[0] if task_manager.get_task_names() else None,
                            label="选择任务"
                        )
                        
                        task_prompt_display = gr.Textbox(
                            label="当前任务提示词",
                            lines=5,
                            interactive=True
                        )
                        
                        # 任务管理按钮
                        with gr.Row():
                            edit_task_btn = gr.Button("✏️ 保存修改", variant="primary")
                            delete_task_btn = gr.Button("🗑️ 删除/重置", variant="secondary")
                        with gr.Row():
                            reload_tasks_btn = gr.Button("🔄 重新加载", variant="secondary")
                        
                        task_status = gr.Textbox(label="任务状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        # 自定义任务
                        gr.Markdown("### 添加自定义任务")
                        new_task_name = gr.Textbox(label="任务名称", placeholder="输入新任务名称")
                        new_task_prompt = gr.Textbox(label="任务提示词", lines=8, placeholder="输入任务提示词")
                        add_task_btn = gr.Button("➕ 添加任务", variant="primary")
                        
                        gr.Markdown("### 任务说明")
                        gr.Markdown("""
                        **默认任务：**
                        - 文本摘要、关键词提取、情感分析等
                        - 可以修改提示词内容
                        - 删除时会重置为原始值
                        
                        **自定义任务：**
                        - 可添加、删除、修改
                        - 支持复杂的提示词模板
                        - 删除时会完全移除
                        
                        **操作说明：**
                        - 选择任务后可直接编辑提示词
                        - 点击"保存修改"应用更改
                        - "删除/重置"对默认任务是重置，对自定义任务是删除
                        """)
            
            # 第三个标签页：文件处理与结果
            with gr.TabItem("📁 文件处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 文件上传
                        gr.Markdown("### 文件上传")
                        file_upload = gr.File(
                            label="上传文件",
                            file_types=[".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".docx", ".doc"]
                        )
                        
                        file_info = gr.Textbox(label="文件信息", interactive=False)
                        
                        # 列选择（仅对表格文件显示）
                        column_dropdown = gr.Dropdown(
                            label="选择要处理的列",
                            visible=False,
                            multiselect=True
                        )
                        
                        # 任务选择
                        gr.Markdown("### 处理任务")
                        selected_task = gr.Dropdown(
                            choices=task_manager.get_task_names(),
                            value=task_manager.get_task_names()[0] if task_manager.get_task_names() else None,
                            label="选择处理任务"
                        )
                        
                        # 处理参数
                        gr.Markdown("### 处理参数")
                        with gr.Row():
                            batch_size = gr.Slider(
                                minimum=1, maximum=50, value=10, step=1,
                                label="批次大小"
                            )
                            max_workers = gr.Slider(
                                minimum=1, maximum=10, value=3, step=1,
                                label="并发数"
                            )
                        
                        # 保存设置
                        gr.Markdown("### 保存设置")
                        save_location = gr.Radio(
                            choices=["当前文件的output目录", "自定义目录"],
                            value="当前文件的output目录",
                            label="保存位置"
                        )
                        
                        custom_save_path = gr.Textbox(
                            label="自定义保存路径",
                            placeholder="输入自定义保存路径...",
                            visible=False
                        )
                        
                        # 处理按钮
                        with gr.Row():
                            process_btn = gr.Button("🔄 开始处理", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ 清除结果", variant="secondary", size="lg")
                    
                    with gr.Column(scale=1):
                        # 文件预览
                        gr.Markdown("### 文件预览")
                        file_preview = gr.Textbox(label="文件内容预览", lines=8, interactive=False)
                        
                        # 处理状态
                        gr.Markdown("### 处理状态")
                        
                        # 添加进度条
                        progress_bar = gr.Progress()
                        processing_progress = gr.Slider(
                            minimum=0, maximum=100, value=0, step=0.1,
                            label="处理进度 (%)",
                            interactive=False,
                            visible=True
                        )
                        
                        process_output = gr.Textbox(label="处理日志", lines=10, interactive=False)
                        
                        # 结果预览
                        gr.Markdown("### 结果预览")
                        result_preview = gr.Textbox(label="处理结果详情", lines=12, interactive=False)
        
        # 显示/隐藏自定义配置
        def toggle_custom_config(preset):
            return gr.Group(visible=(preset == "自定义"))
        
        preset_dropdown.change(
            toggle_custom_config,
            inputs=[preset_dropdown],
            outputs=[custom_config]
        )
        
        # 显示/隐藏自定义保存路径
        save_location.change(
            fn=lambda x: gr.update(visible=(x == "自定义目录")),
            inputs=[save_location],
            outputs=[custom_save_path]
        )
        
        # 加载模型
        load_model_btn.click(
            load_model,
            inputs=[preset_dropdown, custom_name, custom_base_url, custom_api_key, custom_model_name, save_custom_config],
            outputs=[model_status, preset_dropdown]
        )
        
        # 文件上传处理
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[file_info, file_preview, column_dropdown]
        )
        
        # 任务选择时显示提示词
        task_dropdown.change(
            get_task_prompt,
            inputs=[task_dropdown],
            outputs=[task_prompt_display]
        )
        
        # 编辑任务
        edit_task_btn.click(
            edit_task,
            inputs=[task_dropdown, task_prompt_display],
            outputs=[task_status, task_dropdown, selected_task]
        )
        
        # 添加自定义任务
        add_task_btn.click(
            add_custom_task,
            inputs=[new_task_name, new_task_prompt],
            outputs=[task_status, task_dropdown, selected_task]
        )
        
        # 删除任务
        delete_task_btn.click(
            delete_task,
            inputs=[task_dropdown],
            outputs=[task_status, task_dropdown, selected_task]
        )
        
        # 重新加载任务
        reload_tasks_btn.click(
            reload_tasks,
            outputs=[task_status, task_dropdown, selected_task]
        )
        
        # 处理数据
        def process_with_progress(file_upload, column_dropdown, selected_task, batch_size, max_workers, save_location, custom_save_path):
            """带进度更新的处理函数"""
            log, preview, progress = process_data(file_upload, column_dropdown, selected_task, batch_size, max_workers, save_location, custom_save_path)
            return log, preview, progress
        
        # 实时进度更新函数
        def start_processing(file_upload, column_dropdown, selected_task, batch_size, max_workers, save_location, custom_save_path):
            """开始处理并显示实时进度"""
            # 重置进度
            yield "🚀 开始处理...", "", 0.0
            
            # 调用流式处理函数
            try:
                for log, preview, progress in process_data_stream(file_upload, column_dropdown, selected_task, batch_size, max_workers, save_location, custom_save_path):
                    yield log, preview, progress
            except Exception as e:
                yield f"❌ 处理过程中发生错误: {str(e)}", "", 0.0
        
        process_btn.click(
             fn=start_processing,
             inputs=[file_upload, column_dropdown, selected_task, batch_size, max_workers, save_location, custom_save_path],
             outputs=[process_output, result_preview, processing_progress]
         )
         
         # 清除结果功能
        def clear_results():
            return "", "", 0.0
            
        clear_btn.click(
            fn=clear_results,
            outputs=[process_output, result_preview, processing_progress]
        )
    
    return interface

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 创建并启动界面
    interface = create_interface()
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        debug=True
    )
