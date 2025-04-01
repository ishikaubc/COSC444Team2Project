import os
import shutil
DATA_DIR = 'dataset'
def move_first_n_files(src_dir, dst_dir, n=1000):
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取源文件夹内的所有文件（排除子文件夹），按名称排序（可选）
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    files.sort()  # 可选，按名称排序，确保顺序稳定

    # 取前 n 个文件
    files = files[:n]

    # 移动文件
    for file in files:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} → {dst_path}")
source_directory = f"{DATA_DIR}/Non-Vehicles2"  # 替换为源文件夹路径
destination_directory = f"{DATA_DIR}/Non-Vehicles1000"  # 替换为目标文件夹路径
if not os.path.exists(destination_directory):
  os.mkdir(destination_directory)
move_first_n_files(source_directory, destination_directory, n=1000)