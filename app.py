from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form
import dashscope
from dashscope import MultiModalConversation
import base64
import os
import tempfile
import aiofiles
from PIL import Image
import io
from typing import Optional, List
from process import DocumentUploadManager, UploadResult, SearchResult
from rag_service import RAGService, generate_solution
from datetime import datetime
import traceback
import asyncio

# 设置 DashScope API Key
dashscope.api_key = "Your_DashScope_API_Key"

app = FastAPI(title="智能文档处理系统", description="基于AI的图片错误分析和文档索引系统")

# 创建必要的目录
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector", exist_ok=True)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化文档上传管理器
upload_manager = DocumentUploadManager(upload_dir="uploads", vector_base_dir="vector")

# 初始化RAG服务
rag_service = RAGService()

# 全局变量存储查询状态
query_sessions = {}

def cleanup_old_sessions():
    """清理超过10分钟的旧会话"""
    import time
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session in query_sessions.items():
        # 如果会话超过10分钟，标记为清理
        if current_time - session.get('created_at', current_time) > 600:  # 10分钟
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        print(f"清理过期会话: {session_id}")
        del query_sessions[session_id]
    
    return len(sessions_to_remove)

async def analyze_image_content(image_data: bytes):
    """分析图片中的错误信息"""
    try:
        print(f"开始分析图片，图片大小: {len(image_data)} bytes")
        
        # 将图片转换为base64编码
        base64_image = base64.b64encode(image_data).decode('utf-8')
        print("图片base64编码完成")
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "请分析图片中的系统错误信息，提取具体的错误描述并输出。不需要给出解决方案！"
            },
            {
                "role": "user",
                "content": [
                    {"text": "请识别并分析这张图片中的系统错误信息，提取关键的错误描述。不需要给出解决方案！"},
                    {"image": f"data:image/png;base64,{base64_image}"}
                ]
            }
        ]
        
        print("准备调用千问API...")
        # 调用千问视觉模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages,
            temperature=0.1
        )
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            # 提取结果内容
            if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                result = response.output.choices[0]['message']['content']
                first_item = result[0]
                result1 = first_item['text']
                print(f"OCR分析成功: {result1[:100]}...")
                return {"success": True, "result": str(result1)}
            else:
                print("API响应格式异常")
                return {"success": False, "error": "API响应格式异常"}
        else:
            error_msg = getattr(response, 'message', f"状态码: {response.status_code}")
            print(f"API请求失败: {error_msg}")
            return {"success": False, "error": f"API请求失败: {error_msg}"}
            
    except Exception as e:
        print(f"OCR分析异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"分析过程中出现错误: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    auto_search: bool = Form(default=True),
    generate_solution: bool = Form(default=False)
):
    """上传并分析图片，可选择自动检索知识库并生成解决方案"""
    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 读取文件内容
        contents = await file.read()
        
        # 验证图片格式
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="无效的图片格式")
        
        # 分析图片
        analysis_result = await analyze_image_content(contents)
        
        response_data = {
            "filename": file.filename,
            "analysis": analysis_result
        }
        
        # 如果分析成功且启用自动检索，则搜索知识库
        if analysis_result["success"] and auto_search:
            try:
                search_results = upload_manager.search_knowledge_base(
                    query=analysis_result["result"],
                    top_k=5
                )
                
                # 格式化搜索结果
                formatted_results = []
                for result in search_results:
                    formatted_results.append({
                        "content": result.content,
                        "source_file": result.source_file,
                        "score": result.score,
                        "segment_index": result.metadata.get("segment_index", 0)
                    })
                
                response_data["knowledge_search"] = {
                    "success": True,
                    "results": formatted_results,
                    "total_found": len(formatted_results)
                }
                
                # 如果启用解决方案生成且有搜索结果，则生成智能答案
                if generate_solution and search_results:
                    try:
                        rag_response = rag_service.generate_answer_sync(
                            question=analysis_result["result"],
                            search_results=search_results,
                            is_error_analysis=True
                        )
                        
                        response_data["intelligent_solution"] = {
                            "success": rag_response.success,
                            "solution": rag_response.answer if rag_response.success else "",
                            "used_sources": rag_response.used_sources,
                            "confidence": rag_response.confidence,
                            "error": rag_response.error
                        }
                    except Exception as e:
                        response_data["intelligent_solution"] = {
                            "success": False,
                            "solution": "",
                            "used_sources": [],
                            "confidence": 0.0,
                            "error": f"生成解决方案失败: {str(e)}"
                        }
                
            except Exception as e:
                print(f"知识库检索失败: {str(e)}")
                response_data["knowledge_search"] = {
                    "success": False,
                    "error": f"知识库检索失败: {str(e)}"
                }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/search")
async def search_knowledge_base(
    query: str = Form(...),
    file_ids: Optional[str] = Form(default=None),
    top_k: int = Form(default=5)
):
    """搜索知识库"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        # 解析文件ID列表
        file_id_list = None
        if file_ids and file_ids.strip():
            file_id_list = [id.strip() for id in file_ids.split(',') if id.strip()]
        
        # 执行搜索
        search_results = upload_manager.search_knowledge_base(
            query=query.strip(),
            file_ids=file_id_list,
            top_k=min(top_k, 20)  # 限制最大返回数量
        )
        
        # 格式化搜索结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content,
                "source_file": result.source_file,
                "score": result.score,
                "metadata": result.metadata
            })
        
        return JSONResponse(content={
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/knowledge-base/databases")
async def get_available_databases():
    """获取可用的知识库数据库列表"""
    try:
        databases = upload_manager.searcher.get_available_databases()
        return JSONResponse(content={
            "success": True,
            "databases": databases
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据库列表失败: {str(e)}")

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    delimiter: str = Form(default="@@"),
    keep_delimiter: bool = Form(default=False)
):
    """上传并处理文档文件，创建向量索引"""
    try:
        # 读取文件内容
        contents = await file.read()
        
        # 验证文件
        validation_result = upload_manager.validate_file(file.filename, len(contents))
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # 保存上传的文件
        save_result = await upload_manager.save_uploaded_file(contents, file.filename)
        if not save_result["success"]:
            raise HTTPException(status_code=500, detail=save_result["error"])
        
        # 处理文件并创建向量索引
        process_result = upload_manager.process_uploaded_file(
            save_result["file_path"],
            save_result["file_id"],
            save_result["original_filename"],
            delimiter=delimiter,
            keep_delimiter=keep_delimiter
        )
        
        if process_result.success:
            return JSONResponse(content={
                "success": True,
                "message": process_result.message,
                "file_id": process_result.file_id,
                "segments_count": process_result.segments_count,
                "vector_db_path": process_result.vector_db_path
            })
        else:
            raise HTTPException(status_code=500, detail=process_result.error or process_result.message)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/documents")
async def get_documents():
    """获取已处理的文档列表"""
    try:
        records = upload_manager.get_processing_history()
        return JSONResponse(content={
            "success": True,
            "documents": records
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """删除指定的文档和其向量索引"""
    try:
        result = upload_manager.delete_file_and_index(file_id)
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=404, detail=result["error"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "智能文档处理服务正常运行"}

@app.post("/generate-answer")
async def generate_intelligent_answer(
    query: str = Form(...),
    file_ids: Optional[str] = Form(default=None),
    top_k: int = Form(default=5),
    is_error_analysis: bool = Form(default=False)
):
    """基于知识库检索生成智能答案"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        # 解析文件ID列表
        file_id_list = None
        if file_ids and file_ids.strip():
            file_id_list = [id.strip() for id in file_ids.split(',') if id.strip()]
        
        # 执行搜索
        search_results = upload_manager.search_knowledge_base(
            query=query.strip(),
            file_ids=file_id_list,
            top_k=min(top_k, 10)
        )
        
        if not search_results:
            return JSONResponse(content={
                "success": False,
                "error": "未找到相关知识库内容",
                "query": query,
                "results": [],
                "answer": ""
            })
        
        # 生成智能答案
        rag_response = rag_service.generate_answer_sync(
            question=query,
            search_results=search_results,
            is_error_analysis=is_error_analysis
        )
        
        # 格式化搜索结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content,
                "source_file": result.source_file,
                "score": result.score,
                "metadata": result.metadata
            })
        
        return JSONResponse(content={
            "success": rag_response.success,
            "query": query,
            "answer": rag_response.answer if rag_response.success else "",
            "used_sources": rag_response.used_sources,
            "confidence": rag_response.confidence,
            "results": formatted_results,
            "total_search_results": len(search_results),
            "error": rag_response.error
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成答案失败: {str(e)}")

@app.post("/unified-query")
async def unified_query_interface(
    query: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    file_ids: Optional[str] = Form(default=None),
    top_k: int = Form(default=5)
):
    """统一查询接口：支持图片+文字或纯文字查询"""
    try:
        combined_query = query.strip()
        image_analysis_result = None
        
        print(f"统一查询请求: query='{query}', has_file={file is not None}, file_ids='{file_ids}', top_k={top_k}")
        
        # 如果有图片，先分析图片
        if file and file.content_type and file.content_type.startswith("image/"):
            print(f"开始分析图片: {file.filename}, content_type: {file.content_type}")
            try:
                contents = await file.read()
                print(f"图片文件大小: {len(contents)} bytes")
                
                # 验证图片格式
                try:
                    image = Image.open(io.BytesIO(contents))
                    image.verify()
                    print(f"图片验证成功: {image.format}, 尺寸: {image.size}")
                except Exception as img_error:
                    print(f"图片验证失败: {str(img_error)}")
                    raise HTTPException(status_code=400, detail=f"无效的图片格式: {str(img_error)}")
                
                # 分析图片
                analysis_result = await analyze_image_content(contents)
                print(f"图片分析结果: {analysis_result}")
                
                if analysis_result["success"]:
                    image_analysis_result = analysis_result["result"]
                    # 将图片分析结果与用户查询结合
                    if combined_query:
                        combined_query = f"图片分析结果：{analysis_result['result']}\n\n用户问题：{combined_query}"
                    else:
                        combined_query = analysis_result["result"]
                    print(f"组合查询内容: {combined_query[:200]}...")
                else:
                    print(f"图片分析失败: {analysis_result.get('error', '未知错误')}")
                    return JSONResponse(content={
                        "success": False,
                        "error": f"图片分析失败: {analysis_result.get('error', '未知错误')}",
                        "query": query,
                        "has_image": True,
                        "results": [],
                        "answer": ""
                    })
            except Exception as e:
                print(f"图片处理异常: {str(e)}")
                return JSONResponse(content={
                    "success": False,
                    "error": f"图片处理失败: {str(e)}",
                    "query": query,
                    "has_image": True,
                    "results": [],
                    "answer": ""
                })
        
        if not combined_query:
            return JSONResponse(content={
                "success": False,
                "error": "请提供查询内容或上传图片",
                "query": query,
                "has_image": file is not None,
                "results": [],
                "answer": ""
            })
        
        print(f"最终查询内容: {combined_query[:100]}...")
        
        # 解析文件ID列表
        file_id_list = None
        if file_ids and file_ids.strip():
            file_id_list = [id.strip() for id in file_ids.split(',') if id.strip()]
            print(f"指定文件ID: {file_id_list}")
        
        # 执行搜索
        print("开始执行知识库搜索...")
        search_results = upload_manager.search_knowledge_base(
            query=combined_query,
            file_ids=file_id_list,
            top_k=min(top_k, 10)
        )
        print(f"搜索到 {len(search_results)} 个结果")
        
        # 格式化搜索结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content,
                "source_file": result.source_file,
                "score": result.score,
                "metadata": result.metadata
            })
        
        # 生成智能答案
        if search_results:
            print("开始生成智能答案...")
            try:
                rag_response = rag_service.generate_answer_sync(
                    question=combined_query,
                    search_results=search_results,
                    is_error_analysis=True
                )
                
                return JSONResponse(content={
                    "success": rag_response.success,
                    "query": combined_query,
                    "original_query": query,
                    "has_image": image_analysis_result is not None,
                    "image_analysis": image_analysis_result,
                    "answer": rag_response.answer if rag_response.success else "",
                    "used_sources": rag_response.used_sources,
                    "confidence": rag_response.confidence,
                    "results": formatted_results,
                    "total_found": len(search_results),
                    "rag_success": rag_response.success,
                    "rag_error": rag_response.error
                })
            except Exception as e:
                print(f"生成答案失败: {str(e)}")
                return JSONResponse(content={
                    "success": False,
                    "error": f"生成答案失败: {str(e)}",
                    "query": combined_query,
                    "original_query": query,
                    "has_image": image_analysis_result is not None,
                    "image_analysis": image_analysis_result,
                    "results": formatted_results,
                    "answer": ""
                })
        else:
            print("未找到搜索结果")
            return JSONResponse(content={
                "success": False,
                "error": "未找到相关知识库内容",
                "query": combined_query,
                "original_query": query,
                "has_image": image_analysis_result is not None,
                "image_analysis": image_analysis_result,
                "results": [],
                "answer": ""
            })

    except HTTPException:
        raise
    except Exception as e:
        print(f"统一查询异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": f"查询失败: {str(e)}",
            "query": query,
            "has_image": file is not None,
            "results": [],
            "answer": ""
        })

@app.post("/start-query")
async def start_progressive_query():
    """开始分步骤查询"""
    import uuid
    import time
    
    # 清理旧会话
    cleanup_old_sessions()
    
    session_id = str(uuid.uuid4())
    
    query_sessions[session_id] = {
        "status": "started",
        "progress": 0,
        "step": "初始化",
        "cancelled": False,
        "result": None,
        "error": None,
        "created_at": time.time()
    }
    
    print(f"创建新查询会话: {session_id}, 当前会话数: {len(query_sessions)}")
    
    return JSONResponse(content={
        "success": True,
        "session_id": session_id,
        "message": "查询会话已启动"
    })

@app.get("/query-progress/{session_id}")
async def get_query_progress(session_id: str):
    """获取查询进度"""
    if session_id not in query_sessions:
        raise HTTPException(status_code=404, detail="查询会话不存在")
    
    session = query_sessions[session_id]
    
    # 构建响应数据
    response_data = {
        "success": True,
        "session_id": session_id,
        "status": session["status"],
        "progress": session["progress"],
        "step": session["step"],
        "cancelled": session["cancelled"],
        "result": session["result"],
        "error": session["error"]
    }
    
    # 如果有中间的搜索结果，也返回
    if "search_results" in session and session["search_results"]:
        response_data["search_results"] = session["search_results"]
    
    return JSONResponse(content=response_data)

@app.post("/cancel-query/{session_id}")
async def cancel_query(session_id: str):
    """取消查询"""
    if session_id not in query_sessions:
        raise HTTPException(status_code=404, detail="查询会话不存在")
    
    query_sessions[session_id]["cancelled"] = True
    query_sessions[session_id]["status"] = "cancelled"
    query_sessions[session_id]["step"] = "已取消"
    
    print(f"查询会话已取消: {session_id}")
    
    # 延迟清理会话（给前端一点时间获取状态）
    asyncio.create_task(delayed_cleanup_session(session_id))
    
    return JSONResponse(content={
        "success": True,
        "message": "查询已取消"
    })

async def delayed_cleanup_session(session_id: str):
    """延迟清理会话"""
    await asyncio.sleep(2)  # 等待2秒
    if session_id in query_sessions:
        print(f"清理已取消的会话: {session_id}")
        del query_sessions[session_id]

@app.post("/execute-query/{session_id}")
async def execute_progressive_query(
    session_id: str,
    query: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
    file_ids: Optional[str] = Form(default=None),
    top_k: int = Form(default=5)
):
    """启动分步骤查询的异步执行"""
    print(f"=== 启动异步查询会话: {session_id} ===")
    
    if session_id not in query_sessions:
        print(f"查询会话不存在: {session_id}")
        raise HTTPException(status_code=404, detail="查询会话不存在")
    
    session = query_sessions[session_id]
    
    if session["cancelled"]:
        print(f"查询已被取消: {session_id}")
        return JSONResponse(content={
            "success": False,
            "error": "查询已被取消"
        })
    
    # 预处理文件：在异步任务启动前读取文件内容
    file_data = None
    file_info = None
    if file and file.content_type and file.content_type.startswith("image/"):
        try:
            print(f"预读取文件: {file.filename}, {file.content_type}")
            file_data = await file.read()
            file_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(file_data)
            }
            print(f"文件预读取完成: {len(file_data)} bytes")
        except Exception as e:
            print(f"文件预读取失败: {str(e)}")
            return JSONResponse(content={
                "success": False,
                "error": f"文件读取失败: {str(e)}"
            })
    
    # 启动异步任务
    asyncio.create_task(execute_query_async(session_id, query, file_data, file_info, file_ids, top_k))
    
    return JSONResponse(content={
        "success": True,
        "message": "查询任务已启动",
        "session_id": session_id
    })

async def execute_query_async(session_id: str, query: str, file_data: Optional[bytes], file_info: Optional[dict], file_ids: Optional[str], top_k: int):
    """异步执行查询的实际函数"""
    import asyncio  # 在函数内部导入
    
    try:
        if session_id not in query_sessions:
            return
        
        session = query_sessions[session_id]
        
        print(f"=== 开始异步执行查询: {session_id} ===")
        print(f"查询参数: query='{query}', has_file={file_data is not None}, file_ids='{file_ids}', top_k={top_k}")
        
        combined_query = query.strip()
        image_analysis_result = None
        has_image = file_data is not None
        
        print(f"查询类型: has_image={has_image}, combined_query_length={len(combined_query)}")
        
        # 步骤1: 图片OCR处理 (如果有图片)
        if has_image:
            print("=== 步骤1: 开始图片OCR处理 ===")
            session["step"] = "正在进行OCR图片识别..."
            session["progress"] = 10
            
            if session["cancelled"]:
                return
            
            print(f"文件信息: filename={file_info['filename']}, content_type={file_info['content_type']}, size={file_info['size']}")
            
            # 验证图片格式
            try:
                image = Image.open(io.BytesIO(file_data))
                image.verify()
                session["progress"] = 15
                print(f"图片验证成功")
            except Exception as img_error:
                print(f"图片验证失败: {str(img_error)}")
                session["status"] = "error"
                session["error"] = f"无效的图片格式: {str(img_error)}"
                return
            
            # 给前端时间获取进度更新
            await asyncio.sleep(0.5)
            
            # OCR图片分析
            session["step"] = "正在分析图片内容..."
            session["progress"] = 25
            print("开始调用OCR分析...")
            
            if session["cancelled"]:
                return
            
            analysis_result = await analyze_image_content(file_data)
            print(f"OCR分析结果: success={analysis_result['success']}")
            
            if analysis_result["success"]:
                image_analysis_result = analysis_result["result"]
                if combined_query:
                    combined_query = f"图片分析结果：{analysis_result['result']}\n\n用户问题：{combined_query}"
                else:
                    combined_query = analysis_result["result"]
                
                session["step"] = "OCR识别完成，准备检索文档..."
                session["progress"] = 35
                print(f"OCR完成，最终查询长度: {len(combined_query)}")
            else:
                print(f"OCR失败: {analysis_result.get('error', '未知错误')}")
                session["status"] = "error"
                session["error"] = f"图片分析失败: {analysis_result.get('error', '未知错误')}"
                return
        else:
            # 纯文字查询
            print("=== 跳过图片处理，纯文字查询 ===")
            session["step"] = "准备检索文档..."
            session["progress"] = 20
        
        if not combined_query:
            print("错误: 没有查询内容")
            session["status"] = "error"
            session["error"] = "请提供查询内容或上传图片"
            return
        
        # 给前端时间获取进度更新
        await asyncio.sleep(0.5)
        
        # 步骤2: 知识库检索
        print("=== 步骤2: 开始知识库检索 ===")
        session["step"] = "正在搜索知识库..."
        session["progress"] = 50
        
        if session["cancelled"]:
            return
        
        # 解析文件ID列表
        file_id_list = None
        if file_ids and file_ids.strip():
            file_id_list = [id.strip() for id in file_ids.split(',') if id.strip()]
            print(f"指定检索文件: {file_id_list}")
        else:
            print("检索所有文档")
        
        # 执行搜索
        print(f"开始向量搜索，查询: {combined_query[:100]}...")
        try:
            search_results = upload_manager.search_knowledge_base(
                query=combined_query,
                file_ids=file_id_list,
                top_k=min(top_k, 10)
            )
            print(f"搜索完成，找到 {len(search_results)} 个结果")
        except Exception as search_error:
            print(f"知识库搜索失败: {str(search_error)}")
            import traceback
            traceback.print_exc()
            session["status"] = "error"
            session["error"] = f"知识库搜索失败: {str(search_error)}"
            return
        
        if session["cancelled"]:
            return
        
        # 格式化搜索结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content,
                "source_file": result.source_file,
                "score": result.score,
                "metadata": result.metadata
            })
        
        # 更新检索完成状态并存储中间结果
        session["step"] = f"检索文档成功，找到{len(search_results)}个相关文档"
        session["progress"] = 70
        session["search_results"] = formatted_results  # 存储检索结果
        print(f"检索结果已存储到会话")
        
        # 给前端时间获取进度更新和搜索结果
        await asyncio.sleep(1.0)
        
        # 步骤3: 生成智能答案
        if search_results:
            print("=== 步骤3: 准备生成智能答案 ===")
            session["step"] = "准备生成智能解答方案..."
            session["progress"] = 80
            
            if session["cancelled"]:
                return
            
            # 给前端时间获取进度更新
            await asyncio.sleep(0.5)
            
            # 不在这里直接生成答案，而是标记为准备状态
            result = {
                "success": True,
                "query": combined_query,
                "original_query": query,
                "has_image": has_image,
                "image_analysis": image_analysis_result,
                "answer": "",  # 空答案，将通过流式接口生成
                "used_sources": [],
                "confidence": 0.0,
                "results": formatted_results,
                "total_found": len(search_results),
                "rag_success": True,
                "rag_error": None,
                "stream_ready": True  # 标记为可以开始流式生成
            }
            
            # 设置为流式生成状态，而不是完成状态
            session["status"] = "streaming"
            session["step"] = "正在生成智能解答..."
            session["progress"] = 90
            session["result"] = result
            
            print(f"=== 查询进入流式生成阶段: {session_id} ===")
            
        else:
            print("未找到搜索结果")
            result = {
                "success": False,
                "error": "未找到相关知识库内容",
                "query": combined_query,
                "original_query": query,
                "has_image": has_image,
                "image_analysis": image_analysis_result,
                "results": [],
                "answer": "",
                "stream_ready": False
            }
            
            # 完成但没有结果
            session["status"] = "completed"
            session["step"] = "查询完成，未找到相关内容"
            session["progress"] = 100
            session["result"] = result
        
        print(f"=== 异步查询准备完成: {session_id} ===")
        
        # 延迟清理会话（给流式生成足够时间）
        if session["status"] == "streaming":
            # 流式生成状态下不立即清理会话
            pass
        else:
            # 只有在真正完成时才清理
            asyncio.create_task(delayed_cleanup_session(session_id))
        
    except Exception as e:
        if session_id in query_sessions:
            session = query_sessions[session_id]
            session["status"] = "error"
            session["error"] = f"查询失败: {str(e)}"
        print(f"=== 异步查询异常: {session_id} ===")
        print(f"异常信息: {str(e)}")
        import traceback
        traceback.print_exc()

@app.get("/stream-answer/{session_id}")
async def stream_answer(session_id: str):
    """流式返回答案生成过程"""
    from fastapi.responses import StreamingResponse
    import json
    
    print(f"=== 收到流式答案请求: {session_id} ===")
    
    if session_id not in query_sessions:
        print(f"流式请求失败: 查询会话不存在 - {session_id}")
        raise HTTPException(status_code=404, detail="查询会话不存在")
    
    session = query_sessions[session_id]
    print(f"会话状态: {session['status']}")
    print(f"会话结果存在: {'result' in session}")
    
    if session["status"] != "streaming" or not session.get("result"):
        print(f"流式请求失败: 查询未准备好流式生成 - 状态: {session['status']}")
        raise HTTPException(status_code=400, detail="查询未准备好流式生成")
    
    result = session["result"]
    print(f"搜索结果数量: {len(result.get('results', []))}")
    
    # 检查是否有搜索结果
    if not result.get("results") or len(result["results"]) == 0:
        print("流式请求失败: 没有搜索结果")
        # 没有搜索结果，直接返回错误消息
        async def generate_error():
            yield "data: " + json.dumps({"type": "error", "content": "未找到相关知识库内容"}) + "\n\n"
        
        return StreamingResponse(
            generate_error(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    
    # 重建搜索结果对象
    search_results = []
    for r in result["results"]:
        search_result = SearchResult(
            content=r["content"],
            source_file=r["source_file"],
            score=r["score"],
            metadata=r["metadata"]
        )
        search_results.append(search_result)
    
    print(f"重建搜索结果对象完成，共 {len(search_results)} 个")

    async def generate():
        try:
            print("=== 开始流式生成过程 ===")
            
            # 发送开始信号
            yield "data: " + json.dumps({"type": "start", "content": ""}) + "\n\n"
            print("已发送开始信号")
            
            # 判断是否为错误分析
            is_error_analysis = any(keyword in result["query"].lower() for keyword in 
                                  ['错误', '异常', '故障', '问题', 'error', 'exception', '失败'])
            print(f"是否为错误分析: {is_error_analysis}")
            
            # 收集完整答案用于后续URL处理
            full_answer = ""
            used_sources = []
            confidence = 0.0
            chunk_count = 0
            
            # 流式生成答案
            print("开始调用RAG服务流式生成...")
            async for chunk in rag_service.generate_answer_stream(
                question=result["query"],
                search_results=search_results,
                is_error_analysis=is_error_analysis
            ):
                chunk_count += 1
                
                # 确保chunk是字符串类型
                if chunk is None:
                    chunk_str = ""
                else:
                    chunk_str = str(chunk)
                
                print(f"收到第{chunk_count}个chunk: {chunk_str[:50]}..." if len(chunk_str) > 50 else f"收到第{chunk_count}个chunk: {chunk_str}")
                
                # 将chunk累积到完整答案中
                full_answer += chunk_str
                
                # 发送增量chunk（而不是累积的完整答案）
                yield "data: " + json.dumps({"type": "chunk", "content": chunk_str}) + "\n\n"
            
            print(f"RAG流式生成完成，共收到 {chunk_count} 个chunk，总长度: {len(full_answer)}")
            
            # 计算置信度和提取源文件
            if search_results:
                used_sources = list(set(r.source_file for r in search_results))
                avg_similarity = sum(1 - r.score for r in search_results) / len(search_results)
                result_count_factor = min(len(search_results) / 3, 1.0)
                length_factor = min(len(full_answer) / 500, 1.0) if len(full_answer) > 50 else 0.5
                confidence = (avg_similarity * 0.6 + result_count_factor * 0.3 + length_factor * 0.1)
                confidence = min(max(confidence, 0.0), 1.0)
            
            print(f"计算完成 - 置信度: {confidence}, 使用源文件: {used_sources}")
            
            # 发送完成信号，包含完整答案用于URL处理
            yield "data: " + json.dumps({
                "type": "complete", 
                "content": full_answer,
                "used_sources": used_sources,
                "confidence": confidence
            }) + "\n\n"
            
            # 更新会话状态为完成
            if session_id in query_sessions:
                query_sessions[session_id]["status"] = "completed"
                query_sessions[session_id]["step"] = "答案生成完成"
                query_sessions[session_id]["progress"] = 100
                query_sessions[session_id]["result"]["answer"] = full_answer
                query_sessions[session_id]["result"]["used_sources"] = used_sources
                query_sessions[session_id]["result"]["confidence"] = confidence
                print(f"流式生成完成，会话状态已更新: {session_id}")
                
                # 延迟清理会话
                asyncio.create_task(delayed_cleanup_session(session_id))
            
            print("=== 流式生成过程完成 ===")
            
        except Exception as e:
            error_msg = f"流式生成失败: {str(e)}"
            print(f"=== 流式生成异常 ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误消息: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            
            yield "data: " + json.dumps({"type": "error", "content": error_msg}) + "\n\n"
            
            # 更新会话状态为错误
            if session_id in query_sessions:
                query_sessions[session_id]["status"] = "error"
                query_sessions[session_id]["error"] = error_msg
    
    print("返回流式响应")
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/debug/simple-query")
async def debug_simple_query(
    query: str = Form(...),
    file: Optional[UploadFile] = File(default=None)
):
    """调试端点：简单查询，跳过分步机制"""
    try:
        print(f"=== 调试查询开始 ===")
        print(f"查询: {query}")
        print(f"有文件: {file is not None}")
        
        combined_query = query.strip()
        
        # 如果有图片，先分析
        if file and file.content_type and file.content_type.startswith("image/"):
            print("开始图片分析...")
            contents = await file.read()
            print(f"图片大小: {len(contents)} bytes")
            
            analysis_result = await analyze_image_content(contents)
            print(f"图片分析结果: {analysis_result}")
            
            if analysis_result["success"]:
                if combined_query:
                    combined_query = f"图片分析结果：{analysis_result['result']}\n\n用户问题：{combined_query}"
                else:
                    combined_query = analysis_result["result"]
            else:
                return JSONResponse(content={
                    "success": False,
                    "step": "image_analysis",
                    "error": analysis_result.get("error", "图片分析失败")
                })
        
        if not combined_query:
            return JSONResponse(content={
                "success": False,
                "error": "没有查询内容"
            })
        
        print(f"最终查询: {combined_query[:100]}...")
        
        # 知识库检索
        print("开始知识库检索...")
        try:
            search_results = upload_manager.search_knowledge_base(
                query=combined_query,
                top_k=5
            )
            print(f"检索结果: {len(search_results)} 个")
        except Exception as e:
            print(f"检索失败: {str(e)}")
            return JSONResponse(content={
                "success": False,
                "step": "knowledge_search",
                "error": str(e)
            })
        
        # 格式化结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "source_file": result.source_file,
                "score": result.score
            })
        
        # 如果有结果，尝试生成答案
        answer = ""
        if search_results:
            print("开始生成答案...")
            try:
                rag_response = rag_service.generate_answer_sync(
                    question=combined_query,
                    search_results=search_results,
                    is_error_analysis=True
                )
                print(f"RAG结果: success={rag_response.success}")
                if rag_response.success:
                    answer = rag_response.answer
                else:
                    answer = f"RAG生成失败: {rag_response.error}"
            except Exception as e:
                print(f"RAG异常: {str(e)}")
                answer = f"RAG异常: {str(e)}"
        
        print("=== 调试查询完成 ===")
        
        return JSONResponse(content={
            "success": True,
            "query": combined_query,
            "search_results_count": len(search_results),
            "search_results": formatted_results,
            "answer": answer
        })
        
    except Exception as e:
        print(f"调试查询异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "traceback": str(traceback.format_exc())
        })

@app.get("/debug/status")
async def debug_system_status():
    """调试端点：检查系统状态"""
    try:
        status = {
            "timestamp": str(datetime.now()),
            "active_sessions": len(query_sessions),
            "upload_manager": {
                "initialized": upload_manager is not None,
                "upload_dir": upload_manager.upload_dir if upload_manager else None,
                "vector_base_dir": upload_manager.vector_base_dir if upload_manager else None,
            },
            "rag_service": {
                "initialized": rag_service is not None,
            },
            "dashscope_api": {
                "key_configured": bool(dashscope.api_key),
                "key_preview": dashscope.api_key[:10] + "..." if dashscope.api_key else None
            }
        }
        
        # 检查知识库状态
        try:
            databases = upload_manager.searcher.get_available_databases()
            status["knowledge_base"] = {
                "searcher_initialized": True,
                "available_databases": len(databases),
                "databases": databases[:5]  # 只显示前5个
            }
        except Exception as e:
            status["knowledge_base"] = {
                "searcher_initialized": False,
                "error": str(e)
            }
        
        # 检查目录结构
        import os
        status["directories"] = {
            "uploads_exists": os.path.exists("uploads"),
            "vector_exists": os.path.exists("vector"),
            "uploads_count": len(os.listdir("uploads")) if os.path.exists("uploads") else 0,
            "vector_count": len(os.listdir("vector")) if os.path.exists("vector") else 0
        }
        
        return JSONResponse(content={
            "success": True,
            "status": status
        })
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "traceback": str(traceback.format_exc())
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 