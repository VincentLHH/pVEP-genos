# 避免循环导入：不在包级别导入 service，只在需要时从子模块直接导入
#
# 用法（推荐，显式路径）：
#     from api.service import app, run_server
#
# 不要用：
#     from api import app, run_server   # ← 旧的，会触发循环导入
#
# __all__ 保持为空，任何导入都走显式路径
