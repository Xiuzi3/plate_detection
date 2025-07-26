import gradio as gr
import cv2
import torch
import copy
from PIL import Image
import os
import uuid
from detect_plate import load_model, detect_Recognition_plate, draw_result
from plate_recognition.plate_rec import init_model

# 全局变量存储模型
detect_model = None
plate_rec_model = None
device = None

# 创建临时目录用于存储临时图片文件
TEMP_DIR = os.path.join(os.getcwd(), "temp_images")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def initialize_models():
    """初始化模型"""
    global detect_model, plate_rec_model, device

    if detect_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型路径
        detect_model_path = 'weights/plate_detect.pt'
        rec_model_path = 'weights/plate_rec_color.pth'

        # 检查模型文件是否存在
        if not os.path.exists(detect_model_path):
            return False, f"检测模型文件不存在: {detect_model_path}"
        if not os.path.exists(rec_model_path):
            return False, f"识别模型文件不存在: {rec_model_path}"

        try:
            # 加载模型
            detect_model = load_model(detect_model_path, device)
            plate_rec_model = init_model(device, rec_model_path, is_color=True)
            return True, "模型加载成功"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"

    return True, "模型已加载"

def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL格式"""
    if cv2_image is None:
        return None

    # OpenCV使用BGR格式，PIL使用RGB格式
    if len(cv2_image.shape) == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cv2_image)

def vehicle_recognition(input_image):
    """
    车辆识别函数 - 使用本地文件处理方案
    """
    if input_image is None:
        return None, "请先上传图片"

    # 初始化模型
    success, message = initialize_models()
    if not success:
        return None, f"模型初始化失败: {message}"

    temp_path = None
    try:
        # 将PIL图像保存到本地临时文件
        temp_path = save_pil_to_local(input_image)
        if temp_path is None:
            return None, "保存临时图像文件失败"

        # 使用OpenCV直接从文件加载图像（自动为BGR格式）
        cv2_image = load_image_with_opencv(temp_path)
        if cv2_image is None:
            return None, "加载图像文件失败"

        # 复制原图用于绘制结果
        img_copy = copy.deepcopy(cv2_image)

        # 进行车牌检测和识别
        dict_list = detect_Recognition_plate(
            detect_model,
            cv2_image,
            device,
            plate_rec_model,
            img_size=640,
            is_color=True
        )

        # 在图像上绘制结果
        result_image = draw_result(img_copy, dict_list, is_color=True)

        # 转换回PIL格式用于显示
        output_image = cv2_to_pil(result_image)

        # 生成识别结果文本
        result_text = generate_result_text(dict_list)

        return output_image, result_text

    except Exception as e:
        return None, f"识别过程中出现错误: {str(e)}"
    finally:
        # 清理临时文件
        cleanup_temp_file(temp_path)

def generate_result_text(dict_list):
    """生成识别结果文本"""
    if not dict_list:
        return "未检测到车牌"

    result_lines = []
    result_lines.append(f"检测到 {len(dict_list)} 个车牌:")
    result_lines.append("-" * 50)

    for i, result in enumerate(dict_list, 1):
        plate_no = result.get('plate_no', '未识别')
        plate_color = result.get('plate_color', '未知')
        detect_conf = result.get('detect_conf', 0)
        plate_type = result.get('plate_type', 0)

        result_lines.append(f"车牌 {i}:")
        result_lines.append(f"  车牌号: {plate_no}")
        result_lines.append(f" 车牌颜色: {plate_color}")
        result_lines.append(f" 车牌类型: {'双层' if plate_type == 1 else '单层'}")



    return "\n".join(result_lines)

def save_pil_to_local(pil_image):
    """将PIL图像保存到本地临时文件，返回文件路径"""
    if pil_image is None:
        return None

    # 生成唯一的临时文件名
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        # 保存PIL图像到本地文件
        pil_image.save(temp_path, "JPEG", quality=95)
        return temp_path
    except Exception as e:
        print(f"保存临时文件失败: {e}")
        return None

def load_image_with_opencv(image_path):
    """使用OpenCV直接从本地文件加载图像（自动为BGR格式）"""
    if not os.path.exists(image_path):
        return None

    try:
        # OpenCV直接读取为BGR格式，无需转换
        cv2_image = cv2.imread(image_path)
        return cv2_image
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None

def cleanup_temp_file(file_path):
    """清理临时文件"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"清理临时文件失败: {e}")

# 创建Gradio界面

with gr.Blocks(title="车辆识别系统", theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
       <div align="center">
        <h1>车牌识别</h1>
        <p>根据上传的车辆图片，自动识别车牌信息并进行标注。</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  图片上传")
            input_image = gr.Image(
                label="上传图片",
                type="pil",
                height=400
            )
            with gr.Row():
                recognize_btn = gr.Button(
                    "开始识别",
                    variant="primary",
                    size="lg"
                )

                clear_btn = gr.ClearButton(
                    components=[input_image],
                    value="清除图片",
                    variant="primary",
                    size="lg"
                )

            gr.Markdown(
                """
                **使用说明：**
                1. 点击上方区域上传图片
                2. 支持 JPG、PNG、JPEG 格式
                3. 点击"开始识别"按钮
                4. 查看右侧识别结果
                """
            )

        with gr.Column(scale=1):
            gr.Markdown("###  识别结果")
            output_image = gr.Image(
                label="标注结果",
                height=400
            )

            output_text = gr.Textbox(
                label="识别详情",
                lines=8,
                placeholder="识别结果将在这里显示..."
            )

    # 识别图片
    recognize_btn.click(
        fn=vehicle_recognition,
        inputs=[input_image],
        outputs=[output_image, output_text]
    )



    # 示例图片（可选）
    example_images = []
    # 检查imgs目录下的示例图片
    imgs_dir = "imgs"
    if os.path.exists(imgs_dir):
        for img_file in os.listdir(imgs_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                example_images.append([os.path.join(imgs_dir, img_file)])

    if example_images:
        gr.Examples(
            examples=example_images[:6],  # 最多显示6个示例
            inputs=input_image,
            label="示例图片"
        )




if __name__ == "__main__":
    # 创建并启动界面
    demo.launch(
      share=True
    )
