import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import platform

def get_default_font(textSize=20):
    """获取默认字体，支持中文显示"""
    try:
        # 首先尝试加载项目中的字体文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_font_path = os.path.join(os.path.dirname(current_dir), "fonts", "platech.ttf")

        if os.path.exists(project_font_path):
            return ImageFont.truetype(project_font_path, textSize, encoding="utf-8")

        # 根据操作系统选择系统字体
        system = platform.system()

        if system == "Windows":
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
                "C:/Windows/Fonts/arial.ttf",     # Arial
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Arial.ttf",
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]

        # 尝试加载系统字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, textSize, encoding="utf-8")
                except Exception as e:
                    print(f"无法加载字体 {font_path}: {e}")
                    continue

        # 如果所有字体都失败，使用默认字体
        print("使用PIL默认字体")
        return ImageFont.load_default()

    except Exception as e:
        print(f"字体加载失败，使用默认字体: {e}")
        return ImageFont.load_default()

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    """在图像上添加文本，支持中文"""
    try:
        if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)
        fontText = get_default_font(textSize)
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"添加文本失败: {e}")
        # 如果添加文本失败，返回原图
        if isinstance(img, Image.Image):
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return img

if __name__ == '__main__':
    imgPath = "result.jpg"
    img = cv2.imread(imgPath)
    
    saveImg = cv2ImgAddText(img, '中国加油！', 50, 100, (255, 0, 0), 50)
    
    # cv2.imshow('display',saveImg)
    cv2.imwrite('save.jpg',saveImg)
    # cv2.waitKey()