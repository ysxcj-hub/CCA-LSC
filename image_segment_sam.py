# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2
# import os
#
#
# # 加载模型
# sam_checkpoint = "SAM_model/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
#
# device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
#
# mask_generator = SamAutomaticMaskGenerator(sam)
#
# src_folder_path = "data/office-home/Real_World"
# dst_folder_path = "data/office-home/Real_World_mask"
#
# # 遍历文件夹和子文件夹下的所有图片
# def domain_image_mask(src_folder_path, dst_folder_path):  #src_folder_path为原数据集各域的位置,dst_folder_path为目标文件夹的位置
#     for root, dirs, files in os.walk(src_folder_path):
#         for filename in files:
#             if filename.endswith('.jpg'):  # 确保只处理JPG文件
#                 image_path = os.path.join(root, filename)
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 mask = mask_generator.generate(image)  #得到该图像所有掩膜信息的列表
#
#                 first_mask = mask[0]['segmentation']
#                 second_mask = mask[1]['segmentation']
#                 third_mask = mask[2]['segmentation']  #膜的二值化图像
#                 three_d_mask = np.stack([first_mask, second_mask, third_mask], axis=-1)
#
#                 foreground = np.where(three_d_mask, image, 0)
#                 background = np.where(three_d_mask, 0, image)  #分别得到前景和背景
#
#                 rel_path = os.path.relpath(root, src_folder_path)
#                 dst_dir = os.path.join(dst_folder_path, rel_path)
#                 os.makedirs(dst_dir, exist_ok=True)
#                 # mask_path = os.path.join(dst_dir, 'mask_' + filename)
#                 # mask.save(mask_path)
#                 cv2.imwrite('{}/mask_{}.jpg'.format(dst_dir,filename), foreground)
#                 del mask
#                 torch.cuda.empty_cache()
#
# domain_image_mask(src_folder_path,dst_folder_path)

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import gc
from PIL import Image

# 加载模型
sam_checkpoint = "SAM_model/sam_vit_h_4b8939.pth"

# sam_checkpoint = "SAM_model/sam_vit_b_01ec64.pth"
model_type = "vit_h"

device = "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

src_folder_path = "data/office-home/Real_World"
dst_folder_path = "data/office-home/Real_World_mask"


# 遍历文件夹和子文件夹下的所有图片
def domain_image_mask(src_folder_path, dst_folder_path):  # src_folder_path为原数据集各域的位置,dst_folder_path为目标文件夹的位置
    total_files = sum([len(files) for r, d, files in os.walk(src_folder_path)])
    processed_files = 0
    for root, dirs, files in os.walk(src_folder_path):
        for filename in files:
            if filename.endswith('.jpg'):  # 确保只处理JPG文件

                processed_files += 1
                print(f"Processing file {processed_files} of {total_files}: {filename}")

                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image = image.resize((224, 224))  # Resize the image to 224x224 pixels
                image = np.array(image)
                mask = mask_generator.generate(image)  # 得到该图像所有掩膜信息的列表

                rel_path = os.path.relpath(root, src_folder_path)
                dst_dir = os.path.join(dst_folder_path, rel_path)
                os.makedirs(dst_dir, exist_ok=True)

                # if len(mask)>0:
                #     first_mask = mask[0]['segmentation']
                #     three_d_mask = np.stack([first_mask]*3, axis=-1)
                #     background = np.where(three_d_mask, 0, image)  # 分别得到前景和背景
                #     cv2.imwrite('{}/mask_{}'.format(dst_dir, filename), background)
                #
                # else:
                #     cv2.imwrite('{}/mask_{}'.format(dst_dir, filename), image)

                if len(mask)>0:
                    for i in range(len(mask)):
                        first_mask = mask[i]['segmentation']
                        three_d_mask = np.stack([first_mask]*3, axis=-1)
                        background = np.where(three_d_mask, 0, image)  # 分别得到前景和背景
                        foreground = np.where(three_d_mask, image, 0)
                        cv2.imwrite('{}/mask_{}_{}'.format(dst_dir, i, filename), foreground)

                # else:
                #     cv2.imwrite('{}/mask_{}'.format(dst_dir, filename), image)

                # foreground = np.where(three_d_mask, image, 0)

                del mask
                torch.cuda.empty_cache()


                # image_path = os.path.join(root, filename)
                # image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # with torch.no_grad():
                #     mask = mask_generator.generate(image)  # 得到该图像所有掩膜信息的列表
                #
                # first_mask = mask[0]['segmentation']
                # # second_mask = mask[1]['segmentation']
                # # third_mask = mask[2]['segmentation']  # 膜的二值化图像
                # three_d_mask = np.stack([first_mask]*3, axis=-1)
                #
                # background = np.where(three_d_mask, 0, image)  # 分别得到前景和背景
                #
                # rel_path = os.path.relpath(root, src_folder_path)
                # dst_dir = os.path.join(dst_folder_path, rel_path)
                # os.makedirs(dst_dir, exist_ok=True)
                # # mask_path = os.path.join(dst_dir, 'mask_' + filename)
                # # mask.save(mask_path)
                # cv2.imwrite('{}/mask_{}.jpg'.format(dst_dir, filename), background)
                #
                # del mask
                # gc.collect()
                # torch.cuda.empty_cache()
domain_image_mask(src_folder_path, dst_folder_path)

# # 打开原始文件
# with open("data/office-home/txt/Clipart_UT.txt", "r") as infile:
#     lines = infile.readlines()
#
# # 处理每一行
# new_lines = []
# for line in lines:
#     parts = line.split(" ")
#     path_parts = parts[0].split("/")
#     filename = path_parts[-1].split(".")[0]
#     new_path = "/".join(path_parts[:-2]) + "_mask/" + path_parts[-2] + "/mask_" + filename + ".jpg"
#     new_line = new_path + " " + parts[1]
#     new_lines.append(new_line)
#
# # 写入新文件
# with open("data/office-home/txt/Clipart_UT_mask.txt", "w") as outfile:
#     outfile.writelines(new_lines)

# 打开文件a.txt和b.txt
# with open('data/office-home/txt/Product_UT.txt', 'r') as file_a, open('data/office-home/txt/Product_UT_mask.txt', 'r') as file_b:
#     # 读取a.txt和b.txt的内容
#     content_a = file_a.read()
#     content_b = file_b.read()
#
# # 打开文件c.txt
# with open('data/office-home/txt/Product_UT_mask_all.txt', 'w') as file_c:
#     # 将a.txt和b.txt的内容写入c.txt中
#     file_c.write(content_a)
#     file_c.write(content_b)