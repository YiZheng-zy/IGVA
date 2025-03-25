# 确保项目根目录被加入到 sys.path 中
import sys
import os
project_root = '/home/lx/LYX903_balance'
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

from llava.train.train_balance import train 

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
