import numpy as np

def main():
    # 加载矩阵
    try:
        cam2right_base = np.load('./cam2right_base_017322074878.npy')
        cam2left_base = np.load('./cam2left_base_017322074878.npy')
        print("cam2right_base:\n", cam2right_base)
        print("cam2left_base:\n", cam2left_base)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        return
    except Exception as e:
        print(f"错误: 加载矩阵时出错 - {e}")
        return

    # 验证矩阵形状
    if cam2right_base.shape != (4, 4) or cam2left_base.shape != (4, 4):
        print("错误: 矩阵不是4×4的形状")
        return

    # 计算 right_base2left_base
    try:
        right_base2left_base = np.dot(cam2left_base, np.linalg.inv(cam2right_base))
    except np.linalg.LinAlgError:
        print("错误: 矩阵不可逆")
        return

    # 计算 left_base2right_base
    left_base2right_base = np.linalg.inv(right_base2left_base)
    print("right_base2left_base:\n", right_base2left_base)
    print("left_base2right_base:\n", left_base2right_base)

    right_base2left_base = [
        [-1,0,0,-525],
        [0,-1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    left_base2right_base = [
        [-1,0,0,-525],
        [0,-1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

    # 保存结果
    try:
        np.save('right_base2left_base.npy', right_base2left_base)
        np.save('left_base2right_base.npy', left_base2right_base)
        print("成功保存矩阵")
    except Exception as e:
        print(f"错误: 保存矩阵时出错 - {e}")

if __name__ == "__main__":
    main()