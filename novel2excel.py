#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说分段转Excel工具
功能：
1. 支持输入单个小说文件或小说目录
2. 根据指定长度范围进行智能分段
3. 将分段内容保存为Excel文件
4. 支持命令行参数配置

使用示例：
python novel2excel.py -i novel.txt -o output.xlsx --min_len 200 --max_len 800
python novel2excel.py -i novels_dir/ -o output.xlsx --min_len 300 --max_len 1000
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# 尝试导入 chardet（可选）
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    print("警告: 未安装chardet，将使用默认编码检测")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    if HAS_CHARDET:
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)
            r = chardet.detect(raw)
            if r.get('encoding') and r.get('confidence', 0) > 0.7:
                return r['encoding']
        except Exception:
            pass
    
    # 尝试常见编码
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(1000)
            return enc
        except Exception:
            continue
    return 'utf-8'

def clean_text(text: str) -> str:
    """清理文本中的多余空格和空行"""
    if not text:
        return text
    
    # 替换多个连续空格为单个空格（保留换行符前后的空格处理）
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 清理行首行尾空格
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    # 合并多个连续空行为最多两个空行（保持段落分隔）
    cleaned_lines = []
    empty_line_count = 0
    
    for line in lines:
        if line == '':
            empty_line_count += 1
            if empty_line_count <= 2:  # 最多保留两个连续空行
                cleaned_lines.append(line)
        else:
            empty_line_count = 0
            cleaned_lines.append(line)
    
    # 移除开头和结尾的空行
    while cleaned_lines and cleaned_lines[0] == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def read_text(file_path: str, clean_enabled: bool = True) -> str:
    """读取文本文件"""
    enc = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=enc, errors='ignore') as f:
            text = f.read()
        
        # 如果启用文本清理，则进行清理
        if clean_enabled:
            text = clean_text(text)
            logger.debug(f"已清理文件 {Path(file_path).name} 中的多余空格和空行")
        
        return text
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {str(e)}")
        return ""

def split_text(text: str, min_len: int = 200, max_len: int = 800) -> List[str]:
    """智能分段：按双换行分段，合并至指定长度范围"""
    # 按双换行分割段落
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    
    segments = []
    current_segment = ''
    
    for para in paras:
        if not current_segment:
            # 第一个段落
            current_segment = para
        elif len(current_segment) + len(para) + 2 <= max_len:
            # 可以合并到当前段落
            current_segment += "\n\n" + para
        else:
            # 当前段落已满，处理当前段落
            if len(current_segment) < min_len and segments:
                # 当前段落太短，合并到上一个段落
                segments[-1] += "\n\n" + current_segment
            else:
                # 当前段落长度合适，添加到结果
                segments.append(current_segment)
            
            # 开始新段落
            current_segment = para
    
    # 处理最后一个段落
    if current_segment:
        if len(current_segment) < min_len and segments:
            # 最后段落太短，合并到上一个段落
            segments[-1] += "\n\n" + current_segment
        else:
            # 最后段落长度合适，添加到结果
            segments.append(current_segment)
    
    return segments

def get_novel_files(input_path: str) -> List[str]:
    """获取小说文件列表"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() in ['.txt', '.text']:
            return [str(input_path)]
        else:
            logger.warning(f"文件 {input_path} 不是支持的文本格式")
            return []
    
    elif input_path.is_dir():
        # 目录，查找所有txt文件
        novel_files = []
        for ext in ['*.txt', '*.text']:
            novel_files.extend(input_path.glob(ext))
            novel_files.extend(input_path.glob(f"**/{ext}"))  # 递归查找
        
        return [str(f) for f in novel_files]
    
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return []

def process_novels(input_path: str, min_len: int, max_len: int, clean_enabled: bool = True, separate_files: bool = False):
    """处理小说文件，返回分段数据
    
    Args:
        input_path: 输入路径
        min_len: 最小分段长度
        max_len: 最大分段长度
        clean_enabled: 是否启用文本清理
        separate_files: 是否为每个文件单独处理
    
    Returns:
        如果separate_files=False，返回List[Dict]（所有分段的列表）
        如果separate_files=True，返回Dict[str, List[Dict]]（按文件名分组的分段字典）
    """
    novel_files = get_novel_files(input_path)
    
    if not novel_files:
        logger.error("未找到任何小说文件")
        return [] if not separate_files else {}
    
    logger.info(f"找到 {len(novel_files)} 个小说文件")
    if clean_enabled:
        logger.info("已启用文本清理功能（清理多余空格和空行）")
    else:
        logger.info("文本清理功能已禁用")
    
    if separate_files:
        logger.info("模式: 为每个小说文件生成单独的Excel文件")
        segments_by_file = {}
    else:
        logger.info("模式: 将所有小说文件整合到一个Excel文件")
        all_segments = []
    
    for file_path in novel_files:
        logger.info(f"处理文件: {file_path}")
        
        # 读取文件内容
        text = read_text(file_path, clean_enabled)
        if not text.strip():
            logger.warning(f"文件为空或读取失败: {file_path}")
            continue
        
        # 分段
        segments = split_text(text, min_len, max_len)
        logger.info(f"文件 {Path(file_path).name} 分段数量: {len(segments)}")
        
        # 构建分段数据
        file_segments = []
        for i, segment in enumerate(segments, 1):
            segment_data = {
                '文件名': Path(file_path).name,
                '段落序号': i,
                '字符数': len(segment),
                '内容': segment.strip()
            }
            file_segments.append(segment_data)
        
        # 根据模式保存数据
        if separate_files:
            segments_by_file[Path(file_path).name] = file_segments
        else:
            all_segments.extend(file_segments)
    
    return segments_by_file if separate_files else all_segments

def save_to_excel(segments: List[Dict], output_path: str):
    """保存分段数据到Excel文件"""
    if not segments:
        logger.error("没有数据可保存")
        return
    
    try:
        # 创建DataFrame
        df = pd.DataFrame(segments)
        
        # 保存到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='小说分段', index=False)
            
            # 调整列宽
            worksheet = writer.sheets['小说分段']
            worksheet.column_dimensions['A'].width = 20  # 文件名
            worksheet.column_dimensions['B'].width = 10  # 段落序号
            worksheet.column_dimensions['C'].width = 10  # 字符数
            worksheet.column_dimensions['D'].width = 80  # 内容
        
        logger.info(f"成功保存 {len(segments)} 个分段到: {output_path}")
        
    except Exception as e:
        logger.error(f"保存Excel文件失败: {str(e)}")
        raise

def save_segments_by_file(segments_by_file: Dict[str, List[Dict]], output_dir: str):
    """为每个小说文件单独保存Excel文件"""
    if not segments_by_file:
        logger.error("没有数据可保存")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for filename, segments in segments_by_file.items():
        if not segments:
            continue
            
        # 生成输出文件名
        base_name = Path(filename).stem
        output_file = output_dir / f"{base_name}_分段.xlsx"
        
        try:
            # 创建DataFrame（移除文件名列，因为每个文件单独保存）
            df_data = []
            for seg in segments:
                df_data.append({
                    '段落序号': seg['段落序号'],
                    '字符数': seg['字符数'],
                    '内容': seg['内容']
                })
            
            df = pd.DataFrame(df_data)
            
            # 保存到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='小说分段', index=False)
                
                # 调整列宽
                worksheet = writer.sheets['小说分段']
                worksheet.column_dimensions['A'].width = 10  # 段落序号
                worksheet.column_dimensions['B'].width = 10  # 字符数
                worksheet.column_dimensions['C'].width = 80  # 内容
            
            saved_files.append(str(output_file))
            logger.info(f"保存 {filename}: {len(segments)} 个分段 -> {output_file}")
            
        except Exception as e:
            logger.error(f"保存文件 {filename} 失败: {str(e)}")
            continue
    
    logger.info(f"成功保存 {len(saved_files)} 个Excel文件到目录: {output_dir}")
    return saved_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='小说分段转Excel工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python novel2excel.py -i novel.txt -o output.xlsx
  
  # 处理目录，整合到一个Excel文件
  python novel2excel.py -i novels_dir/ -o output.xlsx --min_len 300 --max_len 1000
  
  # 处理目录，为每个小说生成单独的Excel文件
  python novel2excel.py -i novels_dir/ -o output_dir --separate --min_len 200 --max_len 800
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入小说文件或目录路径')
    parser.add_argument('-o', '--output', required=True,
                       help='输出Excel文件路径')
    parser.add_argument('--min_len', type=int, default=200,
                       help='分段最小长度（字符数，默认200）')
    parser.add_argument('--max_len', type=int, default=800,
                       help='分段最大长度（字符数，默认800）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    parser.add_argument('--no-clean', action='store_true',
                       help='禁用文本清理功能（默认启用，清理多余空格和空行）')
    parser.add_argument('--separate', action='store_true',
                       help='为每个小说文件生成单独的Excel文件（仅在输入为目录时有效）')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证参数
    if args.min_len <= 0 or args.max_len <= 0:
        logger.error("分段长度必须大于0")
        sys.exit(1)
    
    if args.min_len >= args.max_len:
        logger.error("最小长度必须小于最大长度")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"开始处理小说文件: {args.input}")
        logger.info(f"分段长度范围: {args.min_len} - {args.max_len} 字符")
        
        # 处理小说文件
        clean_enabled = not args.no_clean  # 默认启用清理，除非指定 --no-clean
        
        # 检查是否使用分离模式
        input_is_dir = Path(args.input).is_dir()
        use_separate = args.separate and input_is_dir
        
        if args.separate and not input_is_dir:
            logger.warning("--separate 参数仅在输入为目录时有效，将使用默认模式")
        
        # 处理文件
        result = process_novels(args.input, args.min_len, args.max_len, clean_enabled, use_separate)
        
        if use_separate:
            # 分离模式：为每个文件生成单独的Excel
            if not result:
                logger.error("没有生成任何分段数据")
                sys.exit(1)
            
            # 如果输出路径是文件，则使用其父目录
            output_path = Path(args.output)
            if output_path.suffix.lower() in ['.xlsx', '.xls']:
                output_dir = output_path.parent / output_path.stem
            else:
                output_dir = output_path
            
            # 保存分离的Excel文件
            saved_files = save_segments_by_file(result, str(output_dir))
            
            # 统计信息
            total_segments = sum(len(segs) for segs in result.values())
            total_chars = sum(seg['字符数'] for segs in result.values() for seg in segs)
            avg_chars = total_chars / total_segments if total_segments else 0
            
            logger.info(f"处理完成！")
            logger.info(f"处理文件数: {len(result)}")
            logger.info(f"生成Excel文件数: {len(saved_files)}")
            logger.info(f"总分段数: {total_segments}")
            logger.info(f"总字符数: {total_chars}")
            logger.info(f"平均分段长度: {avg_chars:.1f} 字符")
            logger.info(f"输出目录: {output_dir}")
            
        else:
            # 整合模式：所有文件整合到一个Excel
            if not result:
                logger.error("没有生成任何分段数据")
                sys.exit(1)
            
            # 保存到Excel
            save_to_excel(result, args.output)
            
            # 统计信息
            total_chars = sum(seg['字符数'] for seg in result)
            avg_chars = total_chars / len(result) if result else 0
            
            logger.info(f"处理完成！")
            logger.info(f"总分段数: {len(result)}")
            logger.info(f"总字符数: {total_chars}")
            logger.info(f"平均分段长度: {avg_chars:.1f} 字符")
            logger.info(f"输出文件: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()