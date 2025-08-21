# -*- coding: utf-8 -*-
"""
小说数据集构建 WebUI（JSONL messages 格式）
- 支持上传TXT文件或指定目录
- 智能分段，按段落合并至阈值
- 助手内容可选择：AI生成 / 使用原文 / 留空
- 可配置 system 提示词与任务类型（写作/续写/修改）
- 内置模型连接和提示词任务管理，独立运行
"""
import os, re, sys, json, time, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import gradio as gr
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 尝试导入 chardet（可选）
try:
    import chardet
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== AI模型配置和客户端 ====================

@dataclass
class ModelConfig:
    """AI模型配置类"""
    name: str
    base_url: str
    api_key: str
    model_name: str
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = 2048
    
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
    def get_preset_configs(cls) -> Dict[str, Dict]:
        """获取预设模型配置"""
        return {
            "DeepSeek": {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "",
                "model_name": "deepseek-chat",
                "timeout": 30,
                "temperature": 0.7
            },
            "OpenAI": {
                "name": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "api_key": "",
                "model_name": "gpt-4o-mini",
                "timeout": 30,
                "temperature": 0.7
            },
            "本地模型": {
                "name": "本地模型",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model_name": "qwen2.5:7b",
                "timeout": 60,
                "temperature": 0.7
            }
        }
    
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
    def save_custom_config(cls, config: Dict):
        """保存自定义模型配置"""
        config_file = "model_configs.json"
        try:
            existing = cls.load_custom_configs()
            existing[config['name']] = config
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存模型配置失败: {str(e)}")
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict]:
        """获取所有配置（预设+自定义）"""
        configs = cls.get_preset_configs()
        configs.update(cls.load_custom_configs())
        return configs
    
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
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError, 
            openai.APITimeoutError,
            openai.APIConnectionError,
            ConnectionError,
            TimeoutError,
            Exception
        ))
    )
    def process_text(self, text: str, prompt: str) -> str:
        """使用AI模型处理文本"""
        start_time = time.time()
        
        try:
            # 检查文本长度，过长的文本进行截断
            if len(text) > 8000:
                text = text[:8000] + "...[文本过长已截断]"
                logger.warning(f"输入文本过长，已截断至8000字符")
            
            messages = [
                {"role": "system", "content": "你是一个专业的文本处理助手，请根据用户的要求处理文本内容。请保持回复简洁明了。"},
                {"role": "user", "content": f"请根据以下要求处理文本：\n\n要求：{prompt}\n\n文本内容：{text}"}
            ]
            
            # 设置合理的max_tokens限制
            max_tokens = self.config.max_tokens
            if max_tokens is None or max_tokens > 4096:
                estimated_tokens = min(len(text), 2048)
                max_tokens = max(estimated_tokens, 256)
            
            max_tokens = min(max_tokens, 2048)
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>", "\n\n\n", "<|im_start|>", "\n\n"],
                timeout=self.config.timeout
            )
            
            result = response.choices[0].message.content.strip()
            
            # 记录处理时间
            elapsed_time = time.time() - start_time
            if elapsed_time > 20:
                logger.warning(f"文本处理耗时较长: {elapsed_time:.2f}秒")
            
            return result
            
        except (openai.APIConnectionError, ConnectionError) as e:
            logger.warning(f"连接错误，正在重试: {str(e)}")
            raise
        except (openai.APITimeoutError, TimeoutError) as e:
            logger.warning(f"请求超时，正在重试: {str(e)}")
            raise
        except openai.RateLimitError as e:
            logger.warning(f"请求频率限制，正在重试: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"AI处理失败: {str(e)}，正在重试")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.process_text("测试", "请回复'连接成功'")
            return "连接成功" in response or "成功" in response
        except Exception:
            return False

# ==================== 提示词任务管理 ====================

class PromptTaskManager:
    """提示词任务管理器"""
    
    def __init__(self):
        self.tasks_file = "prompt_tasks.json"
        self.custom_tasks_file = "custom_tasks.json"
        self._load_tasks()
    
    def _load_tasks(self):
        """加载任务"""
        # 预设任务
        self.predefined_tasks = {
            "小说续写": "请根据上文内容，自然地续写接下来的情节。保持原有的写作风格、人物性格和故事节奏。续写内容应该在300-500字之间，情节要有逻辑性和连贯性。",
            "文本润色": "请对以下文本进行润色和优化，提升语言表达的流畅性、准确性和文学性。保持原意不变，但让表达更加生动、自然。",
            "对话优化": "请优化以下对话内容，使其更加自然、生动，符合人物性格特点。注意对话的节奏感和真实感。",
            "场景描写": "请根据给定的场景要素，创作一段生动的场景描写。注重细节刻画，营造氛围感，字数控制在200-400字。",
            "人物刻画": "请根据给定的人物信息，创作一段人物描写。突出人物的外貌特征、性格特点或心理状态，字数控制在200-300字。",
            "情节构思": "请根据给定的故事背景和要求，构思一个完整的情节发展。包括起因、经过、高潮和结局，逻辑清晰，情节紧凑。"
        }
        
        # 加载自定义任务
        self.custom_tasks = {}
        try:
            if os.path.exists(self.custom_tasks_file):
                with open(self.custom_tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 确保加载的数据是字典格式
                    if isinstance(data, dict):
                        self.custom_tasks = data
                    else:
                        logger.warning(f"自定义任务文件格式错误，已重置为空字典")
                        self.custom_tasks = {}
        except Exception as e:
            logger.warning(f"加载自定义任务失败: {str(e)}")
            self.custom_tasks = {}
    
    def get_all_tasks(self) -> Dict[str, str]:
        """获取所有任务"""
        all_tasks = self.predefined_tasks.copy()
        all_tasks.update(self.custom_tasks)
        return all_tasks
    
    def get_task_names(self) -> List[str]:
        """获取任务名称列表"""
        return list(self.get_all_tasks().keys())
    
    def get_task_prompt(self, task_name: str) -> str:
        """获取任务提示词"""
        all_tasks = self.get_all_tasks()
        return all_tasks.get(task_name, "")
    
    def add_custom_task(self, name: str, prompt: str) -> bool:
        """添加自定义任务"""
        try:
            self.custom_tasks[name] = prompt
            with open(self.custom_tasks_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_tasks, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"添加自定义任务失败: {str(e)}")
            return False
    
    def delete_custom_task(self, name: str) -> bool:
        """删除自定义任务"""
        try:
            if name in self.custom_tasks:
                del self.custom_tasks[name]
                with open(self.custom_tasks_file, 'w', encoding='utf-8') as f:
                    json.dump(self.custom_tasks, f, ensure_ascii=False, indent=2)
                return True
            return False
        except Exception as e:
            logger.error(f"删除自定义任务失败: {str(e)}")
            return False

# ==================== 全局变量 ====================

# 全局变量
current_model_client = None
task_manager = PromptTaskManager()

# ==================== 基本工具函数 ====================

def detect_encoding(file_path: str) -> str:
    if HAS_CHARDET:
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)
            r = chardet.detect(raw)
            if r.get('encoding') and r.get('confidence', 0) > 0.7:
                return r['encoding']
        except Exception:
            pass
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(1000)
            return enc
        except Exception:
            continue
    return 'utf-8'

def read_text(file_path: str) -> str:
    enc = detect_encoding(file_path)
    with open(file_path, 'r', encoding=enc, errors='ignore') as f:
        return f.read()

# 简易分段：按双换行分段，合并至 [min_len, max_len]

def split_text(text: str, min_len: int = 200, max_len: int = 800) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    segs, cur = [], ''
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 2 <= max_len:
            cur += "\n\n" + p
        else:
            if len(cur) < min_len and segs:
                segs[-1] += "\n\n" + cur
            else:
                segs.append(cur)
            cur = p
    if cur:
        if len(cur) < min_len and segs:
            segs[-1] += "\n\n" + cur
        else:
            segs.append(cur)
    return segs

# 构建 messages（保留兼容函数）

def build_messages(segment: str, system_prompt: str, task_type: str, user_extra: str) -> List[Dict[str, str]]:
    """构建messages（兼容函数，新版本中已在build_jsonl中直接构建）"""
    if task_type == '续写':
        user_msg = f"请续写以下段落：\n\n『{segment}』"
    elif task_type == '修改':
        hint = user_extra.strip() or '请优化文笔，使其更流畅、生动且符合中文写作习惯。'
        user_msg = f"请根据以下要求优化这段文本：\n\n要求：{hint}\n\n文本：\n『{segment}』"
    else:  # 写作
        prompt = user_extra.strip() or '请根据给定题材与风格创作一段小说片段（300-500字）。'
        user_msg = prompt
    return [
        {"role": "system", "content": system_prompt.strip() or "你是一名专业的小说创作助手。"},
        {"role": "user", "content": user_msg}
    ]

# AI 生成助手回复

def ai_generate_continuation(client: AIModelClient, segment: str, user_extra: str, selected_task: str = "") -> str:
    """生成续写内容"""
    # 根据选择的任务模板构建提示词
    if selected_task and selected_task != "自定义":
        prompt = task_manager.get_task_prompt(selected_task)
        if not prompt:
            prompt = '请根据上文内容，自然地续写接下来的情节。保持原有的写作风格、人物性格和故事节奏。续写内容应该在300-500字之间，情节要有逻辑性和连贯性。'
    else:
        prompt = '请根据上文内容，自然地续写接下来的情节。保持原有的写作风格、人物性格和故事节奏。续写内容应该在300-500字之间，情节要有逻辑性和连贯性。'
    
    text = segment
    if user_extra.strip():
        text += f"\n\n续写要求：{user_extra.strip()}"
    
    try:
        result = client.process_text(text, prompt)
        if not result or result.strip() == "":
            return f"[AI生成内容为空，请检查模型配置或重试]"
        return result.strip()
    except Exception as e:
        logger.error(f"AI续写生成失败: {str(e)}")
        return f"[AI续写生成失败: {str(e)}]"

def ai_generate_writing(client: AIModelClient, segment: str, user_extra: str, selected_task: str = "") -> str:
    """生成写作内容"""
    # 根据选择的任务模板构建提示词
    if selected_task and selected_task != "自定义":
        prompt = task_manager.get_task_prompt(selected_task)
        if not prompt:
            prompt = '请根据参考文本和创作要求，生成一段高质量的小说内容。保持文风一致，情节合理，字数控制在300-500字。'
    else:
        prompt = '请根据参考文本和创作要求，生成一段高质量的小说内容。保持文风一致，情节合理，字数控制在300-500字。'
    
    requirements = user_extra.strip() or '请根据以下文本内容，创作相关的小说片段。'
    text = f"参考文本：{segment}\n\n创作要求：{requirements}"
    
    try:
        result = client.process_text(text, prompt)
        if not result or result.strip() == "":
            return f"[AI生成内容为空，请检查模型配置或重试]"
        return result.strip()
    except Exception as e:
        logger.error(f"AI写作生成失败: {str(e)}")
        return f"[AI写作生成失败: {str(e)}]"

def ai_generate_modification(client: AIModelClient, segment: str, user_extra: str, selected_task: str = "") -> str:
    """生成修改后的内容"""
    # 根据选择的任务模板构建提示词
    if selected_task and selected_task != "自定义":
        prompt = task_manager.get_task_prompt(selected_task)
        if not prompt:
            prompt = '请对原文进行润色和优化，提升语言表达的流畅性、准确性和文学性。保持原意不变，但让表达更加生动、自然。'
    else:
        prompt = '请对原文进行润色和优化，提升语言表达的流畅性、准确性和文学性。保持原意不变，但让表达更加生动、自然。'
    
    need = user_extra.strip() or '提升文笔与可读性，保持原意不变，优化表达方式。'
    text = f"原文：{segment}\n\n修改要求：{need}"
    
    try:
        result = client.process_text(text, prompt)
        if not result or result.strip() == "":
            return f"[AI生成内容为空，请检查模型配置或重试]"
        return result.strip()
    except Exception as e:
        logger.error(f"AI修改生成失败: {str(e)}")
        return f"[AI修改生成失败: {str(e)}]"

def ai_generate_writing_prompt(client: AIModelClient, assistant_content: str, user_extra: str) -> str:
    """根据生成的内容反推创作要求"""
    try:
        prompt = "请根据以下生成的小说内容，反推出一个合理的创作要求或写作指令，作为用户的提问。要求简洁明了，符合创作逻辑。格式如：'写一个[题材]小说的[场景]，[具体要求]。'"
        text = f"生成的内容：{assistant_content}\n\n用户额外要求：{user_extra}"
        
        result = client.process_text(text, prompt)
        if not result or result.strip() == "":
            return user_extra.strip() or "请创作一段小说内容。"
        return result.strip()
    except Exception as e:
        logger.error(f"生成写作提示词失败: {str(e)}")
        return user_extra.strip() or "请创作一段小说内容。"

def ai_generate_modification_prompt(client: AIModelClient, original_segment: str, modified_content: str, user_extra: str) -> Tuple[str, str]:
    """根据修改后的内容反推原文和修改要求"""
    try:
        # 生成修改要求
        prompt = "请根据原文和修改后的内容，推断出用户可能提出的修改要求。要求简洁明了，具体可操作。"
        text = f"原文：{original_segment}\n\n修改后内容：{modified_content}\n\n用户额外要求：{user_extra}"
        
        modification_request = client.process_text(text, prompt)
        if not modification_request or modification_request.strip() == "":
            modification_request = user_extra.strip() or "请优化文笔，使其更流畅、生动且符合中文写作习惯。"
        
        # 生成稍微简化的原文（模拟未修改前的状态）
        prompt2 = "请将以下文本稍微简化，降低文学性，模拟修改前的原始状态。保持主要内容不变，但表达更朴素一些。"
        original_text = client.process_text(modified_content, prompt2)
        if not original_text or original_text.strip() == "":
            original_text = original_segment
        
        return original_text.strip(), modification_request.strip()
    except Exception as e:
        logger.error(f"生成修改提示词失败: {str(e)}")
        modification_request = user_extra.strip() or "请优化文笔，使其更流畅、生动且符合中文写作习惯。"
        return original_segment, modification_request

# 保留原有的ai_generate函数以兼容其他可能的调用
def ai_generate(client: AIModelClient, task_type: str, segment: str, user_extra: str, selected_task: str = "") -> str:
    """使用AI生成助手回复，基于拆分的原文生成内容（兼容函数）"""
    if task_type == '写作':
        return ai_generate_writing(client, segment, user_extra, selected_task)
    elif task_type == '续写':
        return ai_generate_continuation(client, segment, user_extra, selected_task)
    else:  # 修改
        return ai_generate_modification(client, segment, user_extra, selected_task)

def ai_generate_user_content(client: AIModelClient, task_type: str, assistant_content: str, user_extra: str) -> str:
    """基于assistant内容生成user内容（兼容函数）"""
    if task_type == '写作':
        return ai_generate_writing_prompt(client, assistant_content, user_extra)
    elif task_type == '续写':
        return f"请续写以下段落：\n\n『{assistant_content[:200]}...』"
    else:  # 修改
        original_text, modification_request = ai_generate_modification_prompt(client, assistant_content, assistant_content, user_extra)
        return f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{original_text}』"

def get_default_user_content(task_type: str, user_extra: str) -> str:
    """获取默认用户内容"""
    if task_type == '写作':
        return user_extra.strip() or "请创作一段小说内容。"
    elif task_type == '续写':
        return "请续写以下内容。"
    else:  # 修改
        return user_extra.strip() or "请优化以下文本内容。"

def get_default_prompt(task_type: str) -> str:
    """获取默认提示词"""
    if task_type == '写作':
        return '请根据参考文本和创作要求，生成一段高质量的小说内容。保持文风一致，情节合理，字数控制在300-500字。'
    elif task_type == '续写':
        return '请根据上文内容，自然地续写接下来的情节。保持原有的写作风格、人物性格和故事节奏。续写内容应该在300-500字之间，情节要有逻辑性和连贯性。'
    else:  # 修改
        return '请对原文进行润色和优化，提升语言表达的流畅性、准确性和文学性。保持原意不变，但让表达更加生动、自然。'

# 扫描目录 txt 文件

def scan_dir_txts(dir_path: str) -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        return []
    return [str(fp) for fp in p.rglob('*.txt') if fp.is_file()]

# ==================== 模型管理函数 ====================

def load_model_config(config_name: str, base_url: str = "", api_key: str = "", 
                     model_name: str = "", timeout: int = 30) -> Tuple[str, str]:
    """加载模型配置"""
    global current_model_client
    
    try:
        if config_name == "自定义":
            if not (base_url and api_key and model_name):
                return "❌ 自定义配置需要填写完整的连接信息", ""
            
            config = ModelConfig(
                name="自定义",
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout
            )
        else:
            # 从预设或自定义配置中加载
            all_configs = AIModelClient.get_all_configs()
            if config_name not in all_configs:
                return f"❌ 配置 '{config_name}' 不存在", ""
            
            config_dict = all_configs[config_name]
            if not config_dict.get('api_key'):
                return f"❌ 配置 '{config_name}' 缺少 API Key", ""
            
            config = ModelConfig(
                name=config_dict['name'],
                base_url=config_dict['base_url'],
                api_key=config_dict['api_key'],
                model_name=config_dict['model_name'],
                timeout=config_dict.get('timeout', 30)
            )
        
        # 创建客户端
        current_model_client = AIModelClient(config)
        
        # 测试连接
        logger.info(f"正在测试模型连接: {config.name}")
        if current_model_client.test_connection():
            message = f"✅ 模型 {config.name} 加载成功并连接正常"
            logger.info(message)
            return message, "连接成功"
        else:
            message = f"⚠️ 模型 {config.name} 加载成功但连接测试失败，请检查配置"
            logger.warning(message)
            return message, "连接失败"
            
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}"
        logger.error(error_msg)
        return error_msg, "加载失败"

def save_model_config(name: str, base_url: str, api_key: str, model_name: str, 
                     timeout: int) -> str:
    """保存模型配置"""
    try:
        if not all([name, base_url, api_key, model_name]):
            return "❌ 所有字段都必须填写"
        
        config = {
            "name": name,
            "base_url": base_url,
            "api_key": api_key,
            "model_name": model_name,
            "timeout": timeout,
            "temperature": 0.7
        }
        
        AIModelClient.save_custom_config(config)
        return f"✅ 配置 '{name}' 保存成功"
        
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

# ==================== 任务管理函数 ====================

def get_task_prompt_display(task_name: str) -> str:
    """获取任务提示词用于显示"""
    if not task_name:
        return ""
    return task_manager.get_task_prompt(task_name)

def add_custom_task(name: str, prompt: str) -> Tuple[str, gr.Dropdown]:
    """添加自定义任务"""
    if not name or not prompt:
        return "❌ 任务名称和提示词不能为空", gr.Dropdown()
    
    if task_manager.add_custom_task(name, prompt):
        updated_choices = task_manager.get_task_names() + ["自定义"]
        return f"✅ 任务 '{name}' 添加成功", gr.Dropdown(choices=updated_choices)
    else:
        return "❌ 添加任务失败", gr.Dropdown()

def delete_custom_task(name: str) -> Tuple[str, gr.Dropdown]:
    """删除自定义任务"""
    if not name:
        return "❌ 请选择要删除的任务", gr.Dropdown()
    
    if name in task_manager.predefined_tasks:
        return "❌ 不能删除预设任务", gr.Dropdown()
    
    if task_manager.delete_custom_task(name):
        updated_choices = task_manager.get_task_names() + ["自定义"]
        return f"✅ 任务 '{name}' 删除成功", gr.Dropdown(choices=updated_choices)
    else:
        return "❌ 删除任务失败", gr.Dropdown()

# ==================== 核心处理函数 ====================

def build_jsonl(files: List[str], min_len: int, max_len: int, mode: str, task_type: str,
                system_prompt: str, user_extra: str, selected_task: str = "", 
                progress=gr.Progress()) -> Tuple[str, str]:
    """构建JSONL数据集"""
    global current_model_client
    
    # 修复files为None的错误
    if files is None:
        return None, '未选择有效TXT文件'
    
    all_files = [f for f in files if f and os.path.isfile(f)]
    if not all_files:
        return None, '未选择有效TXT文件'

    # 根据任务类型和模式判断是否需要AI模型
    needs_ai = False
    if mode == 'AI生成':
        needs_ai = True
    elif mode == '使用原文' and task_type == '写作':
        # 写作任务在使用原文模式下也需要AI来反推创作要求
        needs_ai = True
    elif mode == '使用原文' and task_type == '修改':
        # 修改任务在使用原文模式下也需要AI来生成修改要求
        needs_ai = True
    
    if needs_ai and current_model_client is None:
        return None, '当前任务需要AI模型支持，请先加载AI模型'

    # 创建temp目录
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    dataset = []
    total_segments = 0
    
    # 统计总片段数
    for fp in all_files:
        txt = read_text(fp)
        segs = split_text(txt, min_len, max_len)
        total_segments += len(segs)
    
    processed_segments = 0
    
    for fp in all_files:
        txt = read_text(fp)
        segs = split_text(txt, min_len, max_len)
        
        for i, seg in enumerate(segs):
            processed_segments += 1
            progress_ratio = processed_segments / total_segments
            progress(progress_ratio, desc=f"处理 {os.path.basename(fp)} ({i+1}/{len(segs)})")
            
            # 根据任务类型生成不同格式的数据
            if task_type == '续写':
                # 续写任务：使用当前段落作为上文，下一段落作为续写内容
                # 续写任务不需要AI反推，格式相对固定
                if i < len(segs) - 1:  # 确保有下一段
                    current_seg = seg
                    next_seg = segs[i + 1]
                    
                    if mode == 'AI生成':
                        # AI生成续写内容
                        assistant_content = ai_generate_continuation(current_model_client, current_seg, user_extra, selected_task)
                        user_content = f"请续写以下段落：\n\n『{current_seg}』"
                    elif mode == '使用原文':
                        # 使用下一段作为续写内容，不需要AI
                        assistant_content = next_seg
                        user_content = f"请续写以下段落：\n\n『{current_seg}』"
                    else:  # 空白
                        assistant_content = ''
                        user_content = f"请续写以下段落：\n\n『{current_seg}』"
                    
                    msgs = [
                        {"role": "system", "content": system_prompt.strip() or "你是一名专业的小说创作助手，能够根据用户提供的小说上文，自然地续写接下来的情节。"},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                    dataset.append({"messages": msgs})
                    
            elif task_type == '写作':
                 # 写作任务：需要AI反推创作要求（除了空白模式）
                 if mode == 'AI生成':
                     # 先生成创作内容，然后反推创作要求（需要AI）
                     assistant_content = ai_generate_writing(current_model_client, seg, user_extra, selected_task)
                     user_content = ai_generate_writing_prompt(current_model_client, assistant_content, user_extra)
                 elif mode == '使用原文':
                     # 使用原文作为创作内容，需要AI反推创作要求
                     assistant_content = seg
                     if current_model_client:
                         user_content = ai_generate_writing_prompt(current_model_client, assistant_content, user_extra)
                     else:
                         # 如果没有AI模型，使用简单的默认提示
                         user_content = user_extra.strip() or "请创作一段小说内容。"
                 else:  # 空白模式，不需要AI
                     assistant_content = ''
                     user_content = user_extra.strip() or "请创作一段小说内容。"
                 
                 msgs = [
                     {"role": "system", "content": system_prompt.strip() or "你是一名专业的小说创作助手，擅长根据用户的指令生成生动、有趣的故事片段。"},
                     {"role": "user", "content": user_content},
                     {"role": "assistant", "content": assistant_content}
                 ]
                 dataset.append({"messages": msgs})
                
            else:  # 修改任务
                 # 修改任务：需要AI生成修改要求（除了空白模式）
                 if mode == 'AI生成':
                     # 生成修改后的内容，然后反推修改要求（需要AI）
                     assistant_content = ai_generate_modification(current_model_client, seg, user_extra, selected_task)
                     original_text, modification_request = ai_generate_modification_prompt(current_model_client, seg, assistant_content, user_extra)
                     user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{original_text}』"
                 elif mode == '使用原文':
                     # 使用原文作为修改后的内容，需要AI生成修改要求
                     assistant_content = seg
                     if current_model_client:
                         original_text, modification_request = ai_generate_modification_prompt(current_model_client, seg, seg, user_extra)
                         user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{original_text}』"
                     else:
                         # 如果没有AI模型，使用简单的默认修改要求
                         modification_request = user_extra.strip() or "请优化文笔，使其更流畅、生动且符合中文写作习惯。"
                         user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{seg}』"
                 else:  # 空白模式，不需要AI
                     assistant_content = ''
                     modification_request = user_extra.strip() or "请优化文笔，使其更流畅、生动且符合中文写作习惯。"
                     user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{seg}』"
                 
                 msgs = [
                     {"role": "system", "content": system_prompt.strip() or "你是一名专业的编辑助手，擅长根据用户的要求对文字进行润色、修改和优化。"},
                     {"role": "user", "content": user_content},
                     {"role": "assistant", "content": assistant_content}
                 ]
                 dataset.append({"messages": msgs})

    # 保存文件
    out_dir = Path(__file__).parent / '输出数据集'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    task_suffix = selected_task if selected_task and selected_task != "自定义" else task_type
    out_path = out_dir / f'dataset_{task_suffix}_{mode}_{ts}.jsonl'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return str(out_path), f'已生成 {len(dataset)} 条样本，保存到：{out_path}'

# ==================== Gradio UI ====================

def scan_directory(dir_path: str) -> List[str]:
    """扫描目录中的txt文件"""
    if not dir_path or not os.path.exists(dir_path):
        return []
    return scan_dir_txts(dir_path)

def create_ui():
    with gr.Blocks(title='小说数据集构建工具', theme=gr.themes.Ocean()) as demo:
        gr.Markdown('# 📚 小说数据集构建工具\n\n将小说文本转换为 JSONL messages 格式，用于大模型微调')
        
        with gr.Tabs():
            # 主要功能标签页
            with gr.TabItem("📖 数据集构建"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown('### 📁 输入文件')
                        files = gr.File(label='上传TXT文件', file_count='multiple', file_types=['.txt'])
                        dir_path = gr.Textbox(label='或指定目录路径', placeholder='例：D:/novels')
                        scan_btn = gr.Button('📂 扫描目录', variant='secondary')
                        file_list = gr.Textbox(label='待处理文件列表', lines=3, interactive=False)
                        
                        gr.Markdown('### ⚙️ 分段设置')
                        with gr.Row():
                            min_len = gr.Number(label='最小长度', value=200, minimum=50)
                            max_len = gr.Number(label='最大长度', value=800, minimum=100)
                        
                        gr.Markdown('### 🎯 内容生成')
                        mode = gr.Radio(['AI生成', '使用原文', '空白'], label='助手内容来源', value='AI生成')
                        task_type = gr.Radio(['写作', '续写', '修改'], label='任务类型', value='续写')
                        
                        # 任务选择
                        with gr.Row():
                            task_choices = task_manager.get_task_names() + ["自定义"]
                            selected_task = gr.Dropdown(
                                choices=task_choices,
                                label='预设任务模板',
                                value='小说续写' if '小说续写' in task_choices else task_choices[0] if task_choices else "自定义"
                            )
                        
                        task_prompt_display = gr.Textbox(
                            label='当前任务提示词',
                            lines=3,
                            interactive=False,
                            value=task_manager.get_task_prompt('小说续写')
                        )
                        
                        system_prompt = gr.Textbox(
                            label='System 提示词',
                            value='你是一个专业的小说创作助手，擅长各种文学体裁的写作。',
                            lines=2
                        )
                        user_extra = gr.Textbox(
                            label='额外要求（写作主题/修改需求等）',
                            placeholder='例：科幻题材，注重人物心理描写',
                            lines=2
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown('### 🤖 AI模型设置')
                        
                        # 模型配置选择
                        all_configs = AIModelClient.get_all_configs()
                        config_choices = list(all_configs.keys()) + ["自定义"]
                        selected_config = gr.Dropdown(
                            choices=config_choices,
                            label='选择模型配置',
                            value=config_choices[0] if config_choices else "自定义"
                        )
                        
                        # 模型连接参数
                        base_url = gr.Textbox(label='Base URL', value='https://api.deepseek.com/v1')
                        api_key = gr.Textbox(label='API Key', type='password')
                        model_name = gr.Textbox(label='Model Name', value='deepseek-chat')
                        timeout = gr.Number(label='超时时间(秒)', value=30, minimum=10)
                        
                        # 模型操作按钮
                        with gr.Row():
                            load_model_btn = gr.Button('🔗 加载模型', variant='secondary')
                            test_model_btn = gr.Button('🧪 测试连接', variant='secondary')
                        
                        model_status = gr.Textbox(label='模型状态', interactive=False, lines=2)
                        
                        gr.Markdown('### 🚀 执行')
                        build_btn = gr.Button('🔨 开始构建', variant='primary', size='lg')
                        
                        gr.Markdown('### 📊 输出')
                        output_file = gr.File(label='生成的JSONL文件', interactive=False)
                        logs = gr.Textbox(label='处理日志', lines=6, interactive=False)
            
            # 模型管理标签页
            with gr.TabItem("🔧 模型管理"):
                gr.Markdown('### 💾 保存新的模型配置')
                with gr.Row():
                    with gr.Column():
                        new_config_name = gr.Textbox(label='配置名称', placeholder='例：我的DeepSeek')
                        new_base_url = gr.Textbox(label='Base URL', placeholder='https://api.deepseek.com/v1')
                        new_api_key = gr.Textbox(label='API Key', type='password')
                        new_model_name = gr.Textbox(label='Model Name', placeholder='deepseek-chat')
                        new_timeout = gr.Number(label='超时时间(秒)', value=30, minimum=10)
                        
                        save_config_btn = gr.Button('💾 保存配置', variant='primary')
                        save_config_status = gr.Textbox(label='保存状态', interactive=False)
                
                gr.Markdown('### 📋 现有配置列表')
                config_list = gr.Textbox(
                    label='已保存的配置',
                    value='\n'.join([f"• {name}: {config.get('base_url', 'N/A')}" for name, config in all_configs.items()]),
                    lines=8,
                    interactive=False
                )
            
            # 任务管理标签页
            with gr.TabItem("📝 任务管理"):
                gr.Markdown('### ➕ 添加自定义任务')
                with gr.Row():
                    with gr.Column():
                        new_task_name = gr.Textbox(label='任务名称', placeholder='例：角色对话生成')
                        new_task_prompt = gr.Textbox(
                            label='任务提示词',
                            placeholder='请详细描述任务要求...',
                            lines=4
                        )
                        
                        with gr.Row():
                            add_task_btn = gr.Button('➕ 添加任务', variant='primary')
                            delete_task_btn = gr.Button('🗑️ 删除任务', variant='secondary')
                        
                        task_management_status = gr.Textbox(label='操作状态', interactive=False)
                
                gr.Markdown('### 📋 现有任务列表')
                task_list_display = gr.Textbox(
                    label='所有任务',
                    value='\n'.join([f"• {name}: {prompt[:50]}..." for name, prompt in task_manager.get_all_tasks().items()]),
                    lines=10,
                    interactive=False
                )
                
                # 删除任务的下拉选择
                delete_task_dropdown = gr.Dropdown(
                    choices=[name for name in task_manager.get_task_names() if name not in task_manager.predefined_tasks],
                    label='选择要删除的自定义任务',
                    value=None
                )
        
        # ==================== 事件绑定 ====================
        
        # 文件和目录处理
        scan_btn.click(
            fn=lambda path: '\n'.join(scan_directory(path)) if path else '请输入目录路径',
            inputs=[dir_path],
            outputs=[file_list]
        )
        
        files.change(
            fn=lambda fs: '\n'.join([f.name for f in fs]) if fs else '',
            inputs=[files],
            outputs=[file_list]
        )
        
        # 任务选择变化时更新提示词显示
        selected_task.change(
            fn=get_task_prompt_display,
            inputs=[selected_task],
            outputs=[task_prompt_display]
        )
        
        # 模型配置选择变化时更新参数
        def update_model_params(config_name):
            if config_name == "自定义":
                return "", "", "", 30
            
            all_configs = AIModelClient.get_all_configs()
            if config_name in all_configs:
                config = all_configs[config_name]
                return (
                    config.get('base_url', ''),
                    '',  # 不显示保存的API Key
                    config.get('model_name', ''),
                    config.get('timeout', 30)
                )
            return "", "", "", 30
        
        selected_config.change(
            fn=update_model_params,
            inputs=[selected_config],
            outputs=[base_url, api_key, model_name, timeout]
        )
        
        # 模型操作
        load_model_btn.click(
            fn=load_model_config,
            inputs=[selected_config, base_url, api_key, model_name, timeout],
            outputs=[model_status, gr.Textbox(visible=False)]  # 第二个输出用于内部状态
        )
        
        # 保存模型配置
        save_config_btn.click(
            fn=save_model_config,
            inputs=[new_config_name, new_base_url, new_api_key, new_model_name, new_timeout],
            outputs=[save_config_status]
        )
        
        # 任务管理
        add_task_btn.click(
            fn=add_custom_task,
            inputs=[new_task_name, new_task_prompt],
            outputs=[task_management_status, selected_task]
        )
        
        delete_task_btn.click(
            fn=delete_custom_task,
            inputs=[delete_task_dropdown],
            outputs=[task_management_status, selected_task]
        )
        
        # 主要构建功能
        build_btn.click(
            fn=build_jsonl,
            inputs=[files, min_len, max_len, mode, task_type, system_prompt, user_extra, selected_task],
            outputs=[output_file, logs]
        )
    
    return demo

def ui_app():
    return create_ui()

if __name__ == '__main__':
    ui_app().launch()