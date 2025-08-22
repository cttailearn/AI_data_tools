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

class ReversePromptManager:
    """反推提示词管理器"""
    
    def __init__(self):
        self.custom_prompts_file = "custom_reverse_prompts.json"
        self._load_prompts()
    
    def _load_prompts(self):
        """加载反推提示词"""
        # 内置优秀反推提示词模板
        self.predefined_prompts = {
            "小说写作": {
                "基础创作": "请根据以下小说内容，分析其主题、风格、情节特点和写作手法，然后生成一个自然的用户创作请求。用户希望创作具有相似特色的小说片段，请模拟用户可能提出的具体写作需求。",
                "风格模仿": "请仔细分析以下小说片段的写作风格（如叙述视角、语言特色、节奏感、修辞手法等），然后生成一个用户请求，询问如何写出具有相同风格特点的内容。请确保生成的请求具体且实用。",
                "情节构思": "请根据以下小说内容，识别其情节类型、冲突设置、故事结构等特点，然后生成一个用户关于情节创作的具体询问，包括场景设定、人物关系、冲突发展等方面。",
                "人物塑造": "请分析以下小说中的人物刻画手法（如外貌描写、心理描写、对话特点、行为特征等），生成一个用户询问如何塑造类似人物的详细请求。",
                "场景描写": "请分析以下小说中的场景描写技巧（如环境渲染、氛围营造、细节刻画等），生成一个用户询问如何描写类似场景的请求。",
                "对话写作": "请分析以下小说中的对话写作特点（如语言风格、人物性格体现、推动情节等），生成一个用户询问如何写好对话的请求。",
                "情感表达": "请分析以下小说中的情感表达方式（如内心独白、情感渲染、情绪变化等），生成一个用户询问如何表达类似情感的请求。"
            },
            "小说续写": {
                "情节延续": "请根据以下小说前半部分的内容和后半部分的发展，分析情节的逻辑连接点、转折方式和发展脉络，然后生成一个用户续写请求，要求从前半部分自然过渡到后半部分。",
                "风格保持": "请分析以下小说的写作风格、叙述特点和语言特色，生成一个用户请求，询问如何在续写中保持原文的风格一致性，包括语调、节奏、用词习惯等。",
                "情节发展": "请根据小说前文的铺垫和后文的结果，推测中间的情节发展逻辑，生成一个用户询问如何合理发展情节的续写请求。",
                "人物发展": "请分析小说中人物在前后文中的变化和成长，生成一个用户询问如何在续写中展现人物发展的请求。",
                "冲突推进": "请分析小说中冲突的发展脉络，生成一个用户询问如何在续写中推进冲突、制造悬念的请求。",
                "氛围营造": "请分析小说的整体氛围和情绪基调，生成一个用户询问如何在续写中保持或发展这种氛围的请求。"
            },
            "小说修改": {
                "文本润色": "请对比修改前后的文本，分析具体的改进方向（如语言流畅度、表达准确性、文字优美度等），然后生成一个用户关于文本润色的具体请求。",
                "风格调整": "请分析文本修改的风格变化方向（如从平实到华丽、从严肃到轻松等），生成一个用户询问如何调整写作风格的详细请求。",
                "内容完善": "请根据修改后内容的改进点（如增加细节、完善逻辑、丰富情感等），推测用户想要完善的具体方面，生成相应的修改请求。",
                "结构优化": "请分析文本在结构方面的调整（如段落重组、逻辑梳理、层次分明等），生成一个用户询问如何优化文本结构的请求。",
                "语言精炼": "请对比修改前后的语言表达，分析精炼和优化的方向，生成一个用户询问如何让语言更加精炼有力的请求。",
                "情感强化": "请分析修改后文本在情感表达方面的增强，生成一个用户询问如何强化情感表达效果的请求。",
                "逻辑完善": "请分析修改后文本在逻辑方面的完善，生成一个用户询问如何让文本逻辑更加清晰严密的请求。"
            }
        }
        
        # 加载自定义反推提示词
        self.custom_prompts = {}
        try:
            if os.path.exists(self.custom_prompts_file):
                with open(self.custom_prompts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.custom_prompts = data
                    else:
                        logger.warning(f"自定义反推提示词文件格式错误，已重置为空字典")
                        self.custom_prompts = {}
        except Exception as e:
            logger.warning(f"加载自定义反推提示词失败: {str(e)}")
            self.custom_prompts = {}
    
    def get_prompt_categories(self, task_type: str) -> List[str]:
        """获取指定任务类型的提示词分类"""
        categories = []
        if task_type in self.predefined_prompts:
            categories.extend(list(self.predefined_prompts[task_type].keys()))
        if task_type in self.custom_prompts:
            categories.extend(list(self.custom_prompts[task_type].keys()))
        return categories
    
    def get_prompt(self, task_type: str, category: str) -> str:
        """获取指定的反推提示词"""
        # 先查找预设提示词
        if task_type in self.predefined_prompts and category in self.predefined_prompts[task_type]:
            return self.predefined_prompts[task_type][category]
        # 再查找自定义提示词
        if task_type in self.custom_prompts and category in self.custom_prompts[task_type]:
            return self.custom_prompts[task_type][category]
        return ""
    
    def add_custom_prompt(self, task_type: str, category: str, prompt: str) -> bool:
        """添加自定义反推提示词"""
        try:
            if task_type not in self.custom_prompts:
                self.custom_prompts[task_type] = {}
            self.custom_prompts[task_type][category] = prompt
            with open(self.custom_prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_prompts, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"添加自定义反推提示词失败: {str(e)}")
            return False
    
    def delete_custom_prompt(self, task_type: str, category: str) -> bool:
        """删除自定义反推提示词"""
        try:
            if task_type in self.custom_prompts and category in self.custom_prompts[task_type]:
                del self.custom_prompts[task_type][category]
                if not self.custom_prompts[task_type]:  # 如果分类为空，删除整个任务类型
                    del self.custom_prompts[task_type]
                with open(self.custom_prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(self.custom_prompts, f, ensure_ascii=False, indent=2)
                return True
            return False
        except Exception as e:
            logger.error(f"删除自定义反推提示词失败: {str(e)}")
            return False

# ==================== 全局变量 ====================

# 全局变量
current_model_client = None
reverse_prompt_manager = ReversePromptManager()

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

# AI生成助手内容的功能已移除，现在只使用原文内容

def ai_generate_writing_prompt(client: AIModelClient, assistant_content: str, user_extra: str, focus_points: str = "") -> str:
    """根据生成的内容反推创作要求（兼容函数）"""
    return ai_generate_writing_prompt_custom(client, assistant_content, user_extra, focus_points, "")

def ai_generate_writing_prompt_custom(client: AIModelClient, assistant_content: str, user_extra: str, focus_points: str = "", custom_prompt: str = "") -> str:
    """根据生成的内容反推创作要求（支持自定义提示词）"""
    try:
        prompt = custom_prompt.strip() or "请根据以下生成的小说内容，反推出一个合理的创作要求或写作指令，作为用户的提问。要求简洁明了，符合创作逻辑。格式如：'写一个[题材]小说的[场景]，[具体要求]。'"
        
        text = f"生成的内容：{assistant_content}\n\n用户额外要求：{user_extra}"
        if focus_points.strip():
            text += f"\n\n关注点：{focus_points.strip()}"
            if not custom_prompt.strip():  # 只有在使用默认提示词时才添加关注点说明
                prompt += f"\n\n请特别关注以下方面：{focus_points.strip()}"
        
        result = client.process_text(text, prompt)
        if not result or result.strip() == "":
            return user_extra.strip() or "请创作一段小说内容。"
        return result.strip()
    except Exception as e:
        logger.error(f"生成写作提示词失败: {str(e)}")
        return user_extra.strip() or "请创作一段小说内容。"

def ai_generate_modification_prompt(client: AIModelClient, original_segment: str, modified_content: str, user_extra: str, focus_points: str = "") -> Tuple[str, str]:
    """根据修改后的内容反推原文和修改要求（兼容函数）"""
    return ai_generate_modification_prompt_custom(client, original_segment, modified_content, user_extra, focus_points, "")

def ai_generate_modification_prompt_custom(client: AIModelClient, original_segment: str, modified_content: str, user_extra: str, focus_points: str = "", custom_prompt: str = "") -> Tuple[str, str]:
    """根据修改后的内容反推原文和修改要求（支持自定义提示词）"""
    try:
        # 生成修改要求
        prompt = custom_prompt.strip() or "请根据原文和修改后的内容，推断出用户可能提出的修改要求。要求简洁明了，具体可操作。"
        text = f"原文：{original_segment}\n\n修改后内容：{modified_content}\n\n用户额外要求：{user_extra}"
        
        if focus_points.strip():
            text += f"\n\n关注点：{focus_points.strip()}"
            if not custom_prompt.strip():  # 只有在使用默认提示词时才添加关注点说明
                prompt += f"\n\n请特别关注以下方面：{focus_points.strip()}"
        
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

def ai_generate_continuation_prompt_custom(segment: str, user_extra: str, custom_prompt: str = "") -> str:
    """生成续写任务的用户提问内容（支持自定义提示词）"""
    try:
        prompt_template = custom_prompt.strip() or "请续写以下段落："
        
        # 截取前200字符作为展示
        display_segment = segment[:200] + "..." if len(segment) > 200 else segment
        
        if user_extra.strip():
            return f"{prompt_template}\n\n『{display_segment}』\n\n续写要求：{user_extra.strip()}"
        else:
            return f"{prompt_template}\n\n『{display_segment}』"
    except Exception as e:
        logger.error(f"生成续写提示词失败: {str(e)}")
        return f"请续写以下段落：\n\n『{segment[:200]}...』"

# 保留原有的ai_generate函数以兼容其他可能的调用
# AI生成助手内容功能已移除，只使用原文内容

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

def get_random_reverse_prompt(reverse_prompt_manager, task_type: str, category: str) -> str:
    """随机获取反推提示词模板"""
    import random
    
    # 获取指定类别的所有提示词
    prompt = reverse_prompt_manager.get_prompt(task_type, category)
    if not prompt:
        return ""
    
    # 如果提示词包含多个模板（用换行符分隔），随机选择一个
    templates = [template.strip() for template in prompt.split('\n') if template.strip()]
    if templates:
        return random.choice(templates)
    
    return prompt

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

# 反推提示词管理函数
def update_reverse_categories(task_type: str) -> gr.Dropdown:
    """更新反推提示词分类选择"""
    categories = reverse_prompt_manager.get_prompt_categories(task_type)
    return gr.Dropdown(choices=categories, value=categories[0] if categories else None)

def update_reverse_prompt_display(task_type: str, category: str) -> str:
    """更新反推提示词显示"""
    if not task_type or not category:
        return ""
    return reverse_prompt_manager.get_prompt(task_type, category)

def save_reverse_prompt(task_type: str, category: str, prompt: str) -> str:
    """保存反推提示词"""
    if not task_type or not category or not prompt:
        return "❌ 任务类型、分类和提示词不能为空"
    
    # 如果是预设提示词，则添加为自定义提示词
    if reverse_prompt_manager.add_custom_prompt(task_type, category, prompt):
        return f"✅ 反推提示词 '{category}' 保存成功"
    else:
        return "❌ 保存反推提示词失败"

def add_custom_reverse_prompt(task_type: str, category: str, prompt: str) -> Tuple[str, gr.Dropdown]:
    """添加自定义反推提示词"""
    if not task_type or not category or not prompt:
        return "❌ 任务类型、分类名称和提示词不能为空", gr.Dropdown()
    
    if reverse_prompt_manager.add_custom_prompt(task_type, category, prompt):
        # 更新删除选择列表
        custom_choices = []
        for t_type, categories in reverse_prompt_manager.custom_prompts.items():
            for cat in categories.keys():
                custom_choices.append(f"{t_type}:{cat}")
        return f"✅ 自定义反推提示词 '{category}' 添加成功", gr.Dropdown(choices=custom_choices)
    else:
        return "❌ 添加自定义反推提示词失败", gr.Dropdown()

def delete_custom_reverse_prompt(selection: str) -> Tuple[str, gr.Dropdown]:
    """删除自定义反推提示词"""
    if not selection:
        return "❌ 请选择要删除的提示词", gr.Dropdown()
    
    try:
        task_type, category = selection.split(':', 1)
        if reverse_prompt_manager.delete_custom_prompt(task_type, category):
            # 更新删除选择列表
            custom_choices = []
            for t_type, categories in reverse_prompt_manager.custom_prompts.items():
                for cat in categories.keys():
                    custom_choices.append(f"{t_type}:{cat}")
            return f"✅ 自定义反推提示词 '{category}' 删除成功", gr.Dropdown(choices=custom_choices)
        else:
            return "❌ 删除自定义反推提示词失败", gr.Dropdown()
    except ValueError:
        return "❌ 选择格式错误", gr.Dropdown()

def update_template_display(task_type: str) -> str:
    """更新内置模板显示"""
    if task_type in reverse_prompt_manager.predefined_prompts:
        templates = reverse_prompt_manager.predefined_prompts[task_type]
        return '\n'.join([f"• {cat}: {prompt[:80]}..." for cat, prompt in templates.items()])
    return "暂无内置模板"

# ==================== 核心处理函数 ====================

def build_jsonl(dir_path: str, output_dir: str, min_len: int, max_len: int, 
                task_selection: List[str], system_prompt: str, user_extra: str,
                reverse_prompt_type: str, 
                writing_reverse_category: str, continuation_reverse_category: str, modification_reverse_category: str,
                custom_writing_reverse: str, custom_continuation_reverse: str, custom_modification_reverse: str,
                progress=gr.Progress()) -> Tuple[str, str]:
    """构建JSONL数据集"""
    global current_model_client, reverse_prompt_manager
    
    # 固定使用原文模式，AI生成功能已移除
    mode = '使用原文'
    
    # 根据反推提示词类型获取提示词
    if reverse_prompt_type == '内置':
        # 使用内置提示词，支持随机选择
        writing_reverse_prompt = get_random_reverse_prompt(reverse_prompt_manager, '小说写作', writing_reverse_category) if writing_reverse_category else ""
        continuation_reverse_prompt = get_random_reverse_prompt(reverse_prompt_manager, '小说续写', continuation_reverse_category) if continuation_reverse_category else ""
        modification_reverse_prompt = get_random_reverse_prompt(reverse_prompt_manager, '小说修改', modification_reverse_category) if modification_reverse_category else ""
    else:
        # 使用自定义提示词
        writing_reverse_prompt = custom_writing_reverse or ""
        continuation_reverse_prompt = custom_continuation_reverse or ""
        modification_reverse_prompt = custom_modification_reverse or ""
    
    # 检查输入目录
    if not dir_path or not os.path.exists(dir_path):
        return None, '请输入有效的目录路径'
    
    # 扫描目录中的txt文件
    all_files = scan_dir_txts(dir_path)
    if not all_files:
        return None, '目录中未找到TXT文件'
    
    # 检查任务选择
    if not task_selection:
        return None, '请至少选择一个任务类型'

    # 根据任务类型和模式判断是否需要AI模型
    needs_ai = False
    if mode == 'AI生成':
        needs_ai = True
    elif mode == '使用原文':
        # 写作和修改任务在使用原文模式下需要AI来反推要求
        if '写作' in task_selection or '修改' in task_selection:
            needs_ai = True
    
    if needs_ai and current_model_client is None:
        return None, '当前任务需要AI模型支持，请先加载AI模型'

    # 创建输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分批处理配置
    BATCH_SIZE = 100  # 每批处理的样本数量
    
    # 为每个任务类型创建单独的数据集缓存
    datasets_cache = {task: [] for task in task_selection}
    datasets_counters = {task: 0 for task in task_selection}
    
    # 统计总片段数（每个任务类型都要处理）
    total_files = len(all_files)
    total_segments = 0
    file_segments = {}
    
    progress(0.0, desc="正在扫描文件，统计处理量...")
    for fp in all_files:
        txt = read_text(fp)
        segs = split_text(txt, min_len, max_len)
        file_segments[fp] = segs
        total_segments += len(segs) * len(task_selection)
    
    processed_segments = 0
    processed_files = 0
    
    # 创建输出目录和文件句柄
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    # 为每个任务类型创建输出文件
    output_files = {}
    file_handles = {}
    for task_type in task_selection:
        task_suffix = task_type
        out_path = out_dir / f'dataset_{task_suffix}_{mode}_{ts}.jsonl'
        output_files[task_type] = str(out_path)
        file_handles[task_type] = open(out_path, 'w', encoding='utf-8')
    
    progress(0.05, desc=f"扫描完成，共发现 {total_files} 个文件，{total_segments} 个处理任务")
    
    for file_idx, fp in enumerate(all_files):
        segs = file_segments[fp]
        file_name = os.path.basename(fp)
        
        progress(processed_segments / total_segments, desc=f"正在处理文件 {file_idx+1}/{total_files}: {file_name}")
        
        for i, seg in enumerate(segs):
            # 为每个选中的任务类型处理当前段落
            for task_idx, task_type in enumerate(task_selection):
                processed_segments += 1
                progress_ratio = processed_segments / total_segments
                
                # 详细的进度描述
                desc = f"[{processed_segments}/{total_segments}] {file_name} - {task_type} (段落 {i+1}/{len(segs)})"
                progress(progress_ratio, desc=desc)
                
                # 根据任务类型生成不同格式的数据
                if task_type == '续写':
                     # 续写任务：将原文分成两部分，前半部分作为上文，后半部分作为续写内容
                     # 将当前段落分成两部分
                     seg_length = len(seg)
                     if seg_length < 100:  # 段落太短，跳过
                         continue
                     
                     # 找到合适的分割点（尽量在句号、感叹号、问号处分割）
                     split_point = seg_length // 2
                     for punct in ['。', '！', '？', '\n']:
                         # 在中间位置附近寻找标点符号
                         for offset in range(0, seg_length // 4):
                             if split_point + offset < seg_length and seg[split_point + offset] == punct:
                                 split_point = split_point + offset + 1
                                 break
                             if split_point - offset > 0 and seg[split_point - offset] == punct:
                                 split_point = split_point - offset + 1
                                 break
                         else:
                             continue
                         break
                     
                     first_part = seg[:split_point].strip()
                     second_part = seg[split_point:].strip()
                     
                     if len(first_part) < 50 or len(second_part) < 50:  # 分割后部分太短
                         continue
                     
                     # 续写任务：使用AI根据反推提示词生成用户内容
                     assistant_content = second_part
                     user_content = ai_generate_continuation_prompt_custom(first_part, user_extra, continuation_reverse_prompt)
                     
                     msgs = [
                         {"role": "system", "content": system_prompt.strip() or "你是一名专业的小说创作助手，能够根据用户提供的小说上文，自然地续写接下来的情节。"},
                         {"role": "user", "content": user_content},
                         {"role": "assistant", "content": assistant_content}
                     ]
                     
                     # 分批保存机制
                     datasets_cache[task_type].append({"messages": msgs})
                     datasets_counters[task_type] += 1
                     
                     # 当缓存达到批次大小时，写入文件并清空缓存
                     if len(datasets_cache[task_type]) >= BATCH_SIZE:
                         for item in datasets_cache[task_type]:
                             file_handles[task_type].write(json.dumps(item, ensure_ascii=False) + '\n')
                         file_handles[task_type].flush()  # 确保数据写入磁盘
                         datasets_cache[task_type].clear()  # 清空缓存释放内存
                    
                elif task_type == '写作':
                     # 写作任务：使用AI根据反推提示词生成用户内容
                     assistant_content = seg
                     if current_model_client:
                         user_content = ai_generate_writing_prompt_custom(current_model_client, assistant_content, user_extra, user_extra, writing_reverse_prompt)
                     else:
                         # 如果没有AI模型，使用简单的默认提示
                         user_content = user_extra.strip() or "请创作一段小说内容。"
                     
                     msgs = [
                         {"role": "system", "content": system_prompt.strip() or "你是一名专业的小说创作助手，擅长根据用户的指令生成生动、有趣的故事片段。"},
                         {"role": "user", "content": user_content},
                         {"role": "assistant", "content": assistant_content}
                     ]
                     
                     # 分批保存机制
                     datasets_cache[task_type].append({"messages": msgs})
                     datasets_counters[task_type] += 1
                     
                     # 当缓存达到批次大小时，写入文件并清空缓存
                     if len(datasets_cache[task_type]) >= BATCH_SIZE:
                         for item in datasets_cache[task_type]:
                             file_handles[task_type].write(json.dumps(item, ensure_ascii=False) + '\n')
                         file_handles[task_type].flush()  # 确保数据写入磁盘
                         datasets_cache[task_type].clear()  # 清空缓存释放内存
                
                else:  # 修改任务
                     # 修改任务：使用AI根据反推提示词生成用户内容
                     assistant_content = seg
                     if current_model_client:
                         original_text, modification_request = ai_generate_modification_prompt_custom(current_model_client, seg, seg, user_extra, user_extra, modification_reverse_prompt)
                         user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{original_text}』"
                     else:
                         # 如果没有AI模型，使用简单的默认修改要求
                         modification_request = user_extra.strip() or "请优化文笔，使其更流畅、生动且符合中文写作习惯。"
                         user_content = f"请根据以下要求优化这段文本：\n\n要求：{modification_request}\n\n文本：\n『{seg}』"
                     
                     msgs = [
                         {"role": "system", "content": system_prompt.strip() or "你是一名专业的编辑助手，擅长根据用户的要求对文字进行润色、修改和优化。"},
                         {"role": "user", "content": user_content},
                         {"role": "assistant", "content": assistant_content}
                     ]
                     
                     # 分批保存机制
                     datasets_cache[task_type].append({"messages": msgs})
                     datasets_counters[task_type] += 1
                     
                     # 当缓存达到批次大小时，写入文件并清空缓存
                     if len(datasets_cache[task_type]) >= BATCH_SIZE:
                         for item in datasets_cache[task_type]:
                             file_handles[task_type].write(json.dumps(item, ensure_ascii=False) + '\n')
                         file_handles[task_type].flush()  # 确保数据写入磁盘
                         datasets_cache[task_type].clear()  # 清空缓存释放内存

    # 处理剩余的缓存数据并关闭文件
    progress(0.95, desc="正在保存剩余数据...")
    
    total_samples = 0
    final_output_files = []
    
    try:
        for task_type in task_selection:
            # 写入剩余的缓存数据
            if datasets_cache[task_type]:
                for item in datasets_cache[task_type]:
                    file_handles[task_type].write(json.dumps(item, ensure_ascii=False) + '\n')
                file_handles[task_type].flush()
                datasets_cache[task_type].clear()
            
            # 关闭文件句柄
            file_handles[task_type].close()
            
            # 检查文件是否有内容
            if datasets_counters[task_type] > 0:
                final_output_files.append(output_files[task_type])
                total_samples += datasets_counters[task_type]
    
    except Exception as e:
        # 确保所有文件句柄都被关闭
        for handle in file_handles.values():
            try:
                handle.close()
            except:
                pass
        raise e
    
    if not final_output_files:
        progress(1.0, desc="处理完成，但未生成任何样本")
        return None, '未生成任何样本，请检查输入参数'
    
    progress(1.0, desc=f"✅ 处理完成！共生成 {total_samples} 条样本")
    
    # 返回完整的文件路径信息
    file_paths_text = "\n".join([f"📁 {f}" for f in final_output_files])
    return file_paths_text, f'✅ 已生成 {total_samples} 条样本，保存到 {len(final_output_files)} 个文件'

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
                        gr.Markdown('### 📁 输入设置')
                        dir_path = gr.Textbox(label='输入目录路径', placeholder='例：D:/novels')
                        scan_btn = gr.Button('📂 扫描目录', variant='secondary')
                        file_list = gr.Textbox(label='待处理文件列表', lines=3, interactive=False)
                        
                        gr.Markdown('### 💾 输出设置')
                        output_dir = gr.Textbox(
                            label='保存目录',
                            value=str(Path(__file__).parent / '输出数据集'),
                            placeholder='例：D:/output'
                        )
                        
                        gr.Markdown('### ⚙️ 分段设置')
                        with gr.Row():
                            min_len = gr.Number(label='最小长度', value=200, minimum=50)
                            max_len = gr.Number(label='最大长度', value=800, minimum=100)
                        
                        gr.Markdown('### 🎯 任务设置')
                        gr.Markdown('**📋 工作流程说明：**')
                        gr.Markdown('1. **使用原文** → 直接使用小说原文作为助手回复内容\n2. **反推提示词** → 根据助手回复生成用户问题\n3. 最终形成完整的对话数据集')
                        
                        # 任务选择
                        task_selection = gr.CheckboxGroup(
                            choices=['写作', '续写', '修改'],
                            label='选择要执行的任务（可多选）',
                            value=['续写']
                        )
                        
                        gr.Markdown('### 🔄 反推提示词设置')
                        gr.Markdown('**说明：** 反推提示词用于根据助手内容生成用户问题，形成完整的对话数据集')
                        
                        # 反推提示词类型选择
                        reverse_prompt_type = gr.Radio(
                            choices=['内置', '自定义'],
                            label='反推提示词类型',
                            value='内置',
                            info='选择使用内置模板还是自定义提示词'
                        )
                        
                        # 内置提示词设置（所有任务类型通用）
                        with gr.Group(visible=True) as builtin_group:
                            gr.Markdown('**内置提示词设置**')
                            gr.Markdown('*所有任务类型将使用相同的内置反推提示词模板*')
                            
                            # 隐藏的变量，用于兼容性
                            writing_reverse_category = gr.Textbox(value='基础创作', visible=False)
                            continuation_reverse_category = gr.Textbox(value='情节延续', visible=False)
                            modification_reverse_category = gr.Textbox(value='文本润色', visible=False)
                        
                        # 自定义提示词设置（所有任务类型通用）
                        with gr.Group(visible=False) as custom_group:
                            gr.Markdown('**自定义提示词设置**')
                            gr.Markdown('*所有任务类型将使用相同的自定义反推提示词*')
                            
                            # 通用自定义反推提示词
                            custom_universal_reverse = gr.Textbox(
                                label='通用反推提示词',
                                placeholder='请输入用于所有任务类型的反推提示词...',
                                lines=5,
                                value='请根据以下内容，分析其特点和风格，然后生成一个自然的用户请求。用户希望获得具有相似特色的内容，请模拟用户可能提出的具体需求。',
                                visible=True
                            )
                            
                            # 隐藏的变量，用于兼容性
                            custom_writing_reverse = gr.Textbox(visible=False)
                            custom_continuation_reverse = gr.Textbox(visible=False)
                            custom_modification_reverse = gr.Textbox(visible=False)
                        
                        # 注：现在只使用原文内容，AI生成功能已移除
                        
                        system_prompt = gr.Textbox(
                            label='System 提示词',
                            value='你是一个专业的小说创作助手，擅长各种文学体裁的写作。',
                            lines=2
                        )
                        user_extra = gr.Textbox(
                            label='用户需求与关注点',
                            placeholder='例：科幻题材，注重人物心理描写，保持情节连贯性和文笔优美度',
                            lines=3,
                            info='包含写作主题、修改需求、AI反推提示词时的关注点等'
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown('### 🤖 AI模型设置')
                        gr.Markdown('*连接参数配置请在模型管理标签页中完成*')
                        
                        # 模型配置选择
                        all_configs = AIModelClient.get_all_configs()
                        config_choices = list(all_configs.keys())
                        selected_config = gr.Dropdown(
                            choices=config_choices,
                            label='选择模型配置',
                            value=config_choices[0] if config_choices else None,
                            info='请先在模型管理中配置模型'
                        )
                        
                        # 模型操作按钮
                        load_model_btn = gr.Button('🔗 加载并测试模型', variant='primary')
                        
                        model_status = gr.Textbox(label='模型状态', interactive=False, lines=2)
                        
                        gr.Markdown('### 🚀 执行')
                        build_btn = gr.Button('🔨 开始构建', variant='primary', size='lg')
                        
                        gr.Markdown('### 📊 输出')
                        output_file = gr.Textbox(label='生成的文件路径', interactive=False, lines=2)
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
            
            # 反推提示词管理标签页
            with gr.TabItem("🔄 反推提示词管理"):
                gr.Markdown('### 📚 内置反推提示词模板')
                
                # 显示所有内置模板
                with gr.Row():
                    template_task_type = gr.Dropdown(
                        choices=["小说写作", "小说续写", "小说修改"],
                        label='查看任务类型',
                        value='小说写作'
                    )
                
                template_display = gr.Textbox(
                    label='内置模板列表',
                    lines=10,
                    interactive=False,
                    value='\n'.join([f"• {cat}: {prompt[:80]}..." for cat, prompt in reverse_prompt_manager.predefined_prompts.get('小说写作', {}).items()])
                )
                
                gr.Markdown('### ➕ 自定义反推提示词管理')
                with gr.Row():
                    with gr.Column():
                        custom_task_type = gr.Dropdown(
                            choices=["小说写作", "小说续写", "小说修改"],
                            label='任务类型',
                            value='小说写作'
                        )
                        custom_category_name = gr.Textbox(label='分类名称', placeholder='例：情感描写')
                        custom_prompt_content = gr.Textbox(
                            label='反推提示词内容',
                            placeholder='请详细描述反推提示词...',
                            lines=4
                        )
                        
                        with gr.Row():
                            add_custom_prompt_btn = gr.Button('➕ 添加提示词', variant='primary')
                            delete_custom_prompt_btn = gr.Button('🗑️ 删除提示词', variant='secondary')
                        
                        custom_prompt_status = gr.Textbox(label='操作状态', interactive=False)
                
                # 删除自定义提示词的选择
                delete_custom_dropdown = gr.Dropdown(
                    label='选择要删除的自定义提示词',
                    choices=[],
                    value=None
                )

        
        # ==================== 事件绑定 ====================
        
        # 目录处理
        scan_btn.click(
            fn=lambda path: '\n'.join(scan_directory(path)) if path else '请输入目录路径',
            inputs=[dir_path],
            outputs=[file_list]
        )
        
        # 注：反推提示词管理事件绑定已移至专门的标签页
        
        # 反推提示词管理标签页事件
        template_task_type.change(
            fn=update_template_display,
            inputs=[template_task_type],
            outputs=[template_display]
        )
        
        add_custom_prompt_btn.click(
            fn=add_custom_reverse_prompt,
            inputs=[custom_task_type, custom_category_name, custom_prompt_content],
            outputs=[custom_prompt_status, delete_custom_dropdown]
        )
        
        delete_custom_prompt_btn.click(
            fn=delete_custom_reverse_prompt,
            inputs=[delete_custom_dropdown],
            outputs=[custom_prompt_status, delete_custom_dropdown]
        )
        
        # 模型操作 - 简化的加载函数
        def load_selected_model(config_name):
            if not config_name:
                return "❌ 请选择一个模型配置"
            return load_model_config(config_name)[0]  # 只返回状态消息
        
        load_model_btn.click(
            fn=load_selected_model,
            inputs=[selected_config],
            outputs=[model_status]
        )
        
        # 保存模型配置
        save_config_btn.click(
            fn=save_model_config,
            inputs=[new_config_name, new_base_url, new_api_key, new_model_name, new_timeout],
            outputs=[save_config_status]
        )
        

        
        # 反推提示词类型切换事件
        def toggle_reverse_prompt_type(prompt_type):
            if prompt_type == '内置':
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        reverse_prompt_type.change(
            fn=toggle_reverse_prompt_type,
            inputs=[reverse_prompt_type],
            outputs=[builtin_group, custom_group]
        )
        
        # 通用自定义提示词同步事件
        def sync_custom_prompt(universal_prompt):
            return universal_prompt, universal_prompt, universal_prompt
        
        custom_universal_reverse.change(
            fn=sync_custom_prompt,
            inputs=[custom_universal_reverse],
            outputs=[custom_writing_reverse, custom_continuation_reverse, custom_modification_reverse]
        )
        
        # 主要构建功能
        build_btn.click(
            fn=build_jsonl,
            inputs=[dir_path, output_dir, min_len, max_len, task_selection, system_prompt, user_extra,
                   reverse_prompt_type, 
                   writing_reverse_category, continuation_reverse_category, modification_reverse_category,
                   custom_writing_reverse, custom_continuation_reverse, custom_modification_reverse],
            outputs=[output_file, logs]
        )
    
    return demo

def ui_app():
    return create_ui()

if __name__ == '__main__':
    ui_app().launch()
