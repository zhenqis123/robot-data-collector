import cv2
import numpy as np
from pupil_apriltags import Detector

def detect_apriltag_41h12(image_path):
    # 1. 初始化检测器，指定 family 为 tagStandard41h12
    # 这就是之前 OpenCV 识别不到的根本原因
    at_detector = Detector(
        families='tagStandard41h12',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # 2. 读取图像并转为灰度
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 设置相机参数 (从您的 JSON 中提取)
    # [fx, fy, cx, cy]
    camera_params = [1365.706, 1365.269, 983.830, 552.407]
    
    # 设置标签真实物理尺寸 (从 JSON 中提取: 0.0365米)
    tag_size = 0.0365 

    print(f"正在使用 tagStandard41h12 检测图片...")

    # 4. 执行检测
    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )

    print(f"✅ 检测到 {len(tags)} 个标签")

    # 5. 绘制结果
    output_image = img.copy()
    
    for tag in tags:
        tag_id = tag.tag_id
        print(f" - 发现 ID: {tag_id}")
        
        # 提取角点
        corners = tag.corners.astype(int)
        
        # 画绿色的边框
        cv2.polylines(output_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 计算中心点
        center = tag.center.astype(int)
        
        # 在中心写上 ID (红色字，黄色描边)
        text = f"ID:{tag_id}"
        cv2.putText(output_image, text, (center[0]-40, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 4) # 描边
        cv2.putText(output_image, text, (center[0]-40, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   # 内文

        # 画出坐标轴 (可选)
        # pupil_apriltags 返回的 pose_R 和 pose_t
        # 需要手动投影或使用 cv2.drawFrameAxes 辅助（略复杂，此处仅画框确认ID）

    # 6. 显示结果
    # 缩放显示以免图片太大
    screen_res = 1280, 720
    scale = min(screen_res[0] / output_image.shape[1], screen_res[1] / output_image.shape[0])
    if scale < 1:
        new_size = (int(output_image.shape[1] * scale), int(output_image.shape[0] * scale))
        output_image = cv2.resize(output_image, new_size)
    
    cv2.imshow("TagStandard41h12 Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替换您的图片路径
detect_apriltag_41h12('./sample.png')