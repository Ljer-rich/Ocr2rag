import dashscope
from dashscope import MultiModalConversation
import base64
import os
import ast

# 设置 DashScope API Key
dashscope.api_key = "sk-0609abfbca8d4ccb83e86a108da89859"

def analyze_error_image(image_path):
    """直接分析图片中的系统错误信息"""
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件 {image_path} 不存在")
        return
    
    # 将图片转换为base64编码
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"图片读取失败: {e}")
        return
    
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
    
    try:
        # 调用千问视觉模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages,
            temperature=0.1
        )
        
        if response.status_code == 200:
            result = response.output.choices[0]['message']['content']
            print(f"\n图片分析结果：")
            print("=" * 50)
            print(result)
            print("=" * 50)
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.message}")
            
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")

def extract_error_text(ai_result):
    try:
        data = ast.literal_eval(ai_result)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'text' in data[0]:
            return data[0]['text']
    except Exception:
        pass
    return ai_result

if __name__ == "__main__":
    # 分析指定图片
    image_path = "/bigdata/lihuanjia/image/test01.png"  # 可以修改为其他图片路径
    
    print(f"正在分析图片: {image_path}")
    analyze_error_image(image_path) 