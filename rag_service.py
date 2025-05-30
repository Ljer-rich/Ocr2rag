#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 服务模块
负责将检索到的知识传递给大模型，生成问题的解决方案
"""

import json
import requests
import dashscope
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from process import SearchResult


@dataclass
class RAGResponse:
    """RAG响应结果"""
    success: bool
    answer: str
    used_sources: List[str]
    confidence: float
    error: Optional[str] = None


class RAGService:
    """RAG服务类，结合检索结果和大模型生成答案"""
    
    def __init__(self, api_key: str = None):
        """
        初始化RAG服务
        
        Args:
            api_key: DashScope API密钥，如果不提供则使用默认配置
        """
        if api_key:
            dashscope.api_key = api_key
        
        # 配置参数
        self.model_name = "qwen-max"  # 可以根据需要调整模型
        self.max_context_length = 6000  # 最大上下文长度
        self.temperature = 0.3  # 生成温度，较低的值使回答更加精确
        
    def _build_context(self, search_results: List[SearchResult], max_results: int = 5) -> str:
        """
        构建上下文内容
        
        Args:
            search_results: 检索结果列表
            max_results: 最大使用的检索结果数量
        
        Returns:
            格式化的上下文字符串
        """
        if not search_results:
            return "暂无相关知识库内容。"
        
        # 按相似度排序并限制数量
        sorted_results = sorted(search_results, key=lambda x: x.score)[:max_results]
        
        context_parts = []
        for i, result in enumerate(sorted_results, 1):
            similarity = 1 - result.score  # 转换为相似度
            context_part = f"""
【知识片段 {i}】
来源文件: {result.source_file}
相似度: {similarity:.3f}
内容: {result.content}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, is_error_analysis: bool = False) -> str:
        """
        构建发送给大模型的prompt
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            is_error_analysis: 是否为错误分析场景
        
        Returns:
            完整的prompt字符串
        """
        if is_error_analysis:
            prompt_template = """你是一个专业的技术支持专家，擅长分析和解决各种系统错误。

【错误描述】
{question}

【相关知识库内容】
{context}

【任务要求】
1. 仔细分析上述错误描述和相关知识库内容
2. 如果知识库中有相关信息，请优先基于这些信息给出建议
3. 将检索到的url全部完整输出,格式必须为url url url...
4. 不要重复输出url，并确保url有效

请提供专业、准确、易于理解的解决方案："""
        else:
            prompt_template = """你是一个知识渊博的AI助手，能够基于提供的知识库内容回答用户问题。

【用户问题】
{question}

【相关知识库内容】
{context}

【回答要求】
1. 主要基于提供的知识库内容来回答问题
2. 将检索到的url全部完整输出,格式必须为url url url...
3. 回答要准确、详细、条理清晰
4. 不要重复输出url，并确保url有效

请提供专业、准确的回答："""
        
        return prompt_template.format(question=question, context=context)
    
    def _extract_source_files(self, search_results: List[SearchResult]) -> List[str]:
        """提取使用的源文件列表"""
        source_files = []
        for result in search_results:
            if result.source_file not in source_files:
                source_files.append(result.source_file)
        return source_files
    
    def _calculate_confidence(self, search_results: List[SearchResult], answer_length: int) -> float:
        """
        计算答案的置信度
        
        Args:
            search_results: 检索结果列表
            answer_length: 答案长度
        
        Returns:
            置信度分数 (0-1)
        """
        if not search_results:
            return 0.1
        
        # 基于检索结果相似度计算置信度
        avg_similarity = sum(1 - result.score for result in search_results) / len(search_results)
        
        # 基于结果数量的调整
        result_count_factor = min(len(search_results) / 3, 1.0)
        
        # 基于答案长度的调整（适中长度的答案通常更可靠）
        length_factor = min(answer_length / 500, 1.0) if answer_length > 50 else 0.5
        
        # 综合计算置信度
        confidence = (avg_similarity * 0.6 + result_count_factor * 0.3 + length_factor * 0.1)
        return min(max(confidence, 0.0), 1.0)
    
    async def generate_answer(self, 
                            question: str, 
                            search_results: List[SearchResult],
                            is_error_analysis: bool = False) -> RAGResponse:
        """
        基于检索结果生成答案
        
        Args:
            question: 用户问题
            search_results: 检索结果列表
            is_error_analysis: 是否为错误分析场景
        
        Returns:
            RAG响应结果
        """
        try:
            # 构建上下文
            context = self._build_context(search_results)
            
            # 构建prompt
            prompt = self._build_prompt(question, context, is_error_analysis)
            
            # 调用千问API
            response = await self._call_qwen_api(prompt)
            
            if response["success"]:
                # 提取源文件信息
                used_sources = self._extract_source_files(search_results)
                
                # 计算置信度
                confidence = self._calculate_confidence(search_results, len(response["answer"]))
                
                return RAGResponse(
                    success=True,
                    answer=response["answer"],
                    used_sources=used_sources,
                    confidence=confidence
                )
            else:
                return RAGResponse(
                    success=False,
                    answer="",
                    used_sources=[],
                    confidence=0.0,
                    error=response["error"]
                )
                
        except Exception as e:
            return RAGResponse(
                success=False,
                answer="",
                used_sources=[],
                confidence=0.0,
                error=f"生成答案时发生错误: {str(e)}"
            )
    
    async def _call_qwen_api(self, prompt: str) -> Dict[str, Any]:
        """
        调用千问API
        
        Args:
            prompt: 发送给模型的prompt
        
        Returns:
            API响应结果
        """
        try:
            # 使用DashScope调用千问模型
            from dashscope import Generation
            
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=2000,  # 最大生成token数
                top_p=0.8,
                repetition_penalty=1.1
            )
            
            print(f"Qwen API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                # 提取生成的文本
                if hasattr(response, 'output') and hasattr(response.output, 'text'):
                    answer = response.output.text.strip()
                    return {
                        "success": True,
                        "answer": answer
                    }
                else:
                    return {
                        "success": False,
                        "error": "API响应格式异常"
                    }
            else:
                error_msg = getattr(response, 'message', f"状态码: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API请求失败: {error_msg}"
                }
                
        except Exception as e:
            print(f"调用千问API时发生错误: {str(e)}")
            return {
                "success": False,
                "error": f"API调用异常: {str(e)}"
            }
    
    def generate_answer_sync(self, 
                           question: str, 
                           search_results: List[SearchResult],
                           is_error_analysis: bool = False) -> RAGResponse:
        """
        同步版本的答案生成（用于非异步环境）
        
        Args:
            question: 用户问题
            search_results: 检索结果列表
            is_error_analysis: 是否为错误分析场景
        
        Returns:
            RAG响应结果
        """
        try:
            # 构建上下文
            context = self._build_context(search_results)
            
            # 构建prompt
            prompt = self._build_prompt(question, context, is_error_analysis)
            
            # 调用千问API（同步版本）
            response = self._call_qwen_api_sync(prompt)
            
            if response["success"]:
                # 提取源文件信息
                used_sources = self._extract_source_files(search_results)
                
                # 计算置信度
                confidence = self._calculate_confidence(search_results, len(response["answer"]))
                
                return RAGResponse(
                    success=True,
                    answer=response["answer"],
                    used_sources=used_sources,
                    confidence=confidence
                )
            else:
                return RAGResponse(
                    success=False,
                    answer="",
                    used_sources=[],
                    confidence=0.0,
                    error=response["error"]
                )
                
        except Exception as e:
            return RAGResponse(
                success=False,
                answer="",
                used_sources=[],
                confidence=0.0,
                error=f"生成答案时发生错误: {str(e)}"
            )
    
    def _call_qwen_api_sync(self, prompt: str) -> Dict[str, Any]:
        """
        同步调用千问API
        
        Args:
            prompt: 发送给模型的prompt
        
        Returns:
            API响应结果
        """
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=2000,
                top_p=0.8,
                repetition_penalty=1.1
            )
            
            print(f"Qwen API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                if hasattr(response, 'output') and hasattr(response.output, 'text'):
                    answer = response.output.text.strip()
                    return {
                        "success": True,
                        "answer": answer
                    }
                else:
                    return {
                        "success": False,
                        "error": "API响应格式异常"
                    }
            else:
                error_msg = getattr(response, 'message', f"状态码: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API请求失败: {error_msg}"
                }
                
        except Exception as e:
            print(f"调用千问API时发生错误: {str(e)}")
            return {
                "success": False,
                "error": f"API调用异常: {str(e)}"
            }

    async def generate_answer_stream(self, 
                                   question: str, 
                                   search_results: List[SearchResult],
                                   is_error_analysis: bool = False):
        """
        基于检索结果流式生成答案
        
        Args:
            question: 用户问题
            search_results: 检索结果列表
            is_error_analysis: 是否为错误分析场景
        
        Yields:
            流式响应的字符串片段
        """
        try:
            # 构建上下文
            context = self._build_context(search_results)
            
            # 构建prompt
            prompt = self._build_prompt(question, context, is_error_analysis)
            
            # 调用千问API流式接口
            async for chunk in self._call_qwen_api_stream(prompt):
                yield chunk
                
        except Exception as e:
            yield f"生成答案时出错: {str(e)}"
    
    async def _call_qwen_api_stream(self, prompt: str):
        """
        调用千问API的流式接口
        
        Args:
            prompt: 发送给模型的prompt
            
        Yields:
            流式响应的字符串片段
        """
        try:
            from dashscope import Generation
            
            print(f"开始流式调用千问API...")
            print(f"Prompt长度: {len(prompt)}")
            print(f"使用模型: {self.model_name}")
            
            # 尝试使用messages格式
            try:
                messages = [
                    {'role': 'user', 'content': prompt}
                ]
                
                print("尝试使用messages格式调用API...")
                responses = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=2000,
                    top_p=0.8
                )
            except Exception as msg_error:
                print(f"messages格式调用失败: {str(msg_error)}")
                print("尝试使用prompt格式调用API...")
                
                # 如果messages格式失败，尝试使用prompt格式
                responses = Generation.call(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=2000,
                    top_p=0.8
                )
            
            print(f"流式API调用启动成功")
            chunk_count = 0
            previous_content = ""  # 用于计算增量内容
            
            for response in responses:
                chunk_count += 1
                print(f"收到第{chunk_count}个响应块，状态码: {response.status_code}")
                
                if response.status_code == 200:
                    current_full_content = None
                    
                    # 尝试多种方式获取内容
                    if hasattr(response, 'output'):
                        output = response.output
                        
                        # 方式1: choices格式
                        if hasattr(output, 'choices') and output.choices and len(output.choices) > 0:
                            choice = output.choices[0]
                            if isinstance(choice, dict):
                                if 'message' in choice and 'content' in choice['message']:
                                    current_full_content = choice['message']['content']
                                elif 'text' in choice:
                                    current_full_content = choice['text']
                            elif hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                current_full_content = choice.message.content
                        
                        # 方式2: 直接text属性
                        elif hasattr(output, 'text'):
                            current_full_content = output.text
                        
                        # 方式3: 其他可能的属性
                        elif hasattr(output, 'content'):
                            current_full_content = output.content
                    
                    if current_full_content:
                        # 计算增量内容（千问API返回的是累积内容）
                        if len(current_full_content) > len(previous_content):
                            incremental_content = current_full_content[len(previous_content):]
                            print(f"增量内容({len(incremental_content)}字符): {incremental_content[:50]}...")
                            previous_content = current_full_content
                            yield incremental_content
                        else:
                            print(f"无新增量内容，跳过此响应块")
                    else:
                        print(f"空内容响应，响应结构: {response}")
                        # 不yield空内容，继续等待下一个chunk
                else:
                    error_msg = f"API错误状态码: {response.status_code}"
                    if hasattr(response, 'message'):
                        error_msg += f", 错误信息: {response.message}"
                    print(error_msg)
                    yield error_msg
                    break
            
            print(f"流式生成完成，总共收到{chunk_count}个响应块，最终内容长度: {len(previous_content)}")
            
            # 如果没有收到任何内容，返回错误信息
            if chunk_count == 0:
                yield "未收到API响应"
            elif not previous_content:
                yield "API未返回任何内容"
                    
        except Exception as e:
            error_msg = f"调用千问API流式接口时出错: {str(e)}"
            print(error_msg)
            import traceback
            print(f"详细错误堆栈: {traceback.format_exc()}")
            yield error_msg


# 便捷函数
def generate_solution(question: str, 
                     search_results: List[SearchResult], 
                     is_error_analysis: bool = False,
                     api_key: str = None) -> RAGResponse:
    """
    便捷函数：生成问题解决方案
    
    Args:
        question: 用户问题
        search_results: 检索结果列表
        is_error_analysis: 是否为错误分析场景
        api_key: API密钥
    
    Returns:
        RAG响应结果
    """
    rag_service = RAGService(api_key)
    return rag_service.generate_answer_sync(question, search_results, is_error_analysis)


# 测试用例
if __name__ == "__main__":
    # 创建测试数据
    from process import SearchResult
    
    test_results = [
        SearchResult(
            content="系统错误代码500通常表示服务器内部错误，可能是由于配置问题或程序bug导致的。",
            metadata={"segment_index": 1, "file_name": "错误代码手册.txt"},
            score=0.2,
            source_file="错误代码手册.txt"
        ),
        SearchResult(
            content="解决500错误的常用方法：1. 检查服务器日志 2. 重启相关服务 3. 检查配置文件",
            metadata={"segment_index": 2, "file_name": "故障排除指南.txt"},
            score=0.15,
            source_file="故障排除指南.txt"
        )
    ]
    
    # 测试RAG服务
    rag_service = RAGService()
    
    test_question = "我遇到了HTTP 500错误，应该如何解决？"
    
    result = rag_service.generate_answer_sync(
        question=test_question,
        search_results=test_results,
        is_error_analysis=True
    )
    
    print("=== RAG测试结果 ===")
    print(f"成功: {result.success}")
    print(f"答案: {result.answer}")
    print(f"使用源文件: {result.used_sources}")
    print(f"置信度: {result.confidence:.3f}")
    if result.error:
        print(f"错误: {result.error}") 