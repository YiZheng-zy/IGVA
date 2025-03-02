# 确保项目根目录被加入到 sys.path 中
# import sys
# import os
# project_root = '/'#项目路径
# if project_root not in sys.path: 
#     sys.path.insert(0, project_root)
from llava.train.train import train

#print('11111111111111111111111111111111111')

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
