from pathlib import Path
import shutil
from tqdm import tqdm

def delete_depth_folders(root: str, max_depth: int):
    root_path = Path(root)
    if not root_path.exists():
        print(f"错误: 路径 {root} 不存在")
        return

    all_dirs_to_delete = []
    
    # 1. 扫描阶段：递归寻找符合条件的文件夹
    def traverse_directory(path: Path, current_depth: int):
        # 严格限制：如果当前深度已经超过 max_depth，直接返回
        if current_depth > max_depth:
            return
        
        try:
            # 遍历当前目录下的子项
            for item in path.iterdir():
                if item.is_dir():
                    # 如果找到了目标文件夹 'depth'
                    if item.name == 'depth':
                        all_dirs_to_delete.append(item)
                        # 优化：既然要删除这个文件夹，就没必要再进入它的子目录了
                        continue 
                    
                    # 只有当还没达到 max_depth 时，才继续向深层递归
                    if current_depth < max_depth:
                        traverse_directory(item, current_depth + 1)
        except PermissionError:
            print(f"权限拒绝: 无法访问 {path}")
        except Exception as e:
            print(f"访问 {path} 时出错: {e}")

    print(f"正在扫描目录 (最大深度: {max_depth})...")
    traverse_directory(root_path, 0)
    
    count = len(all_dirs_to_delete)
    if count == 0:
        print("未发现符合条件的 'depth' 文件夹。")
        return

    print(f"找到 {count} 个待删除文件夹。")

    # 2. 删除阶段：使用 tqdm 显示进度
    # 使用 set() 确保路径唯一（防止父子目录重复包含，虽然上面的逻辑已规避）
    for dirpath in tqdm(all_dirs_to_delete, desc="正在执行删除", unit="dir"):
        try:
            shutil.rmtree(dirpath)
            # tqdm.write 用于在不破坏进度条的情况下打印日志
            tqdm.write(f"已删除: {dirpath}")
        except Exception as e:
            tqdm.write(f"删除失败 {dirpath}: {e}")

if __name__ == "__main__":
    # 配置参数
    root_directory = "/media/zwp/PHILIPS/captures/2025-12-24" 
    max_depth = 2  
    
    delete_depth_folders(root_directory, max_depth)