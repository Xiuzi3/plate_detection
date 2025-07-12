from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import time

# 导入项目中的检测模块
from detect_plate import detect_Recognition_plate, draw_result, load_model
from plate_recognition.plate_rec import init_model, cv_imread

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 全局变量存储模型
detect_model = None
plate_rec_model = None
device = None

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_models():
    """初始化模型"""
    global detect_model, plate_rec_model, device

    # 模型路径
    detect_model_path = 'weights/plate_detect.pt'
    rec_model_path = 'weights/plate_rec_color.pth'

    if not os.path.exists(detect_model_path):
        print(f"检测模型文件不存在: {detect_model_path}")
        return False

    if not os.path.exists(rec_model_path):
        print(f"识别模型文件不存在: {rec_model_path}")
        return False

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 初始化检测模型
        detect_model = load_model(detect_model_path, device)

        # 初始化识别模型
        plate_rec_model = init_model(device, rec_model_path, is_color=True)

        print("模型初始化成功")
        return True
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return False

def process_image(image_path):
    """处理图片进行车牌识别"""
    try:
        # 读取图片
        img = cv_imread(image_path)
        if img is None:
            return None, "无法读取图片"

        # 如果是4通道图片，转换为3通道
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 获取检测结果
        dict_list = detect_Recognition_plate(
            detect_model, img, device, plate_rec_model,
            img_size=640, is_color=True
        )

        # 解析结果
        plate_results = []

        for result in dict_list:
            plate_info = {
                'plate_no': result['plate_no'],
                'confidence': float(result['detect_conf']),
                'color': result.get('plate_color', 'unknown'),
                'bbox': result['rect'],
                'plate_type': '双层' if result['plate_type'] == 1 else '单层'
            }
            plate_results.append(plate_info)

        # 绘制结果图片
        result_img = draw_result(img, dict_list, is_color=True)

        # 保存结果图片
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)

        return plate_results, result_filename

    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None, str(e)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 添加时间戳避免文件名���突
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 处理图片
        plate_results, result_filename = process_image(filepath)

        if plate_results is not None:
            return jsonify({
                'success': True,
                'plates': plate_results,
                'result_image': f'/static/results/{result_filename}',
                'original_image': f'/static/uploads/{filename}'
            })
        else:
            return jsonify({'error': f'处理失败: {result_filename}'})

    return jsonify({'error': '不支持的文件格式'})

@app.route('/static/<path:filename>')
@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    response = send_file(os.path.join('static', filename))
    # 禁用缓存，确保每次都获取最新文件
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response



if __name__ == '__main__':
    # 初始化模型
    print("正在初始化模型...")
    if init_models():
        print("模型初始化成功！")
        print("启动Flask应用...")
        print("=" * 50)
        print("服务器启动成功！")
        print("请在浏览器中访问: http://localhost:5000")
        print("按 Ctrl+C 停止服务器")
        print("=" * 50)
        try:
            app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
        except Exception as e:
            print(f"启动应用时出错: {e}")
            print("请检查端口5000是否被其他程序占用")
    else:
        print("模型初始化失败，无法启动应用")
        print("请检查模型文件是否存在并且格式正确")
