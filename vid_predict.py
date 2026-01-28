# 多帧模型的图像可视化

import cv2
import numpy as np
from PIL import Image

from vid_map_coco import get_history_imgs
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.TASA import Tasanet

from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression

class Pred_vid(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'D:/affair/college/ISTD/TMP/logs/loss_2025_12_11_11_26_21/best_epoch_weights.pth',
        "classes_path"      : 'D:/affair/college/ISTD/TMP/model_data/classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net    = Tasanet(self.num_classes, num_frame=5)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
                
     #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, images, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(images[0])[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        images       = [cvtColor(image) for image in images]
        c_image = images[-1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]
        # (3, 640, 640) -> (3, 16, 640, 640)
        image_data = np.stack(image_data, axis=1)
        
        image_data  = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            outputs = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return c_image

            top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
            top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes   = outputs[0][:, :4]

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * c_image.size[1] + 15).astype('int32'))  #######
        thickness   = int(max((c_image.size[0] + c_image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(c_image.size[1], np.floor(bottom).astype('int32'))
                right   = min(c_image.size[0], np.floor(right).astype('int32'))
                
                # dir_save_path = "img_crop"
                dir_save_path =os.path.join(r'D:\affair\college\ISTD\TMP\output', "img_crop")
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = c_image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(c_image.size[1], np.floor(bottom).astype('int32'))
            right   = min(c_image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(c_image)
            # label_size = draw.textsize(label, font)
            label_size = draw.textbbox((125, 20),label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size[:2])], fill=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin)], fill=self.colors[c])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return c_image
    
if __name__ == "__main__":
    yolo = Pred_vid()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示序列图片预测，输出为视频
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#


    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        # while True:
            # img = input('Input image filename:')
        img = 'D:\\affair\\college\\ISTD\\TMP\\DAUB\\images\\test\\data6\\364.bmp'
        try:
            img = get_history_imgs(img)
            images = [Image.open(item) for item in img]
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(images, crop = crop, count=count)
            output_dir = r'D:\affair\college\ISTD\TMP\output'
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, "Our_68_100.jpg") # 保持原文件名
            r_image.save("Our_68_100.jpg")
            print("Detect finish.")
    elif mode == "video":
        import numpy as np
        from tqdm import tqdm
        dir_path = 'D:\\affair\\college\\ISTD\\TMP\\DAUB\\images\\test\\data6'
        output_dir = r'D:\affair\college\ISTD\TMP\output'
        os.makedirs(output_dir, exist_ok=True) 
        output_video_path = os.path.join(output_dir, "output.avi")
        images = os.listdir(dir_path)
        images.sort(key=lambda x:int(x[:-4]))
        list_img = []
        # for image in tqdm(images):
        #     image = dir_path+image
        #     img = get_history_imgs(image)
        #     imgs = [Image.open(item) for item in img]
        #     r_image = yolo.detect_image(imgs, crop = crop, count=count)
        #     list_img.append(cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR))
        for image_name in tqdm(images):
            # -------------------------------------------------------------
            # 🌟 修复点 2 (关键): 使用 os.path.join() 正确拼接路径
            # -------------------------------------------------------------
            image_full_path = os.path.join(dir_path, image_name)
            
            # 使用正确的完整路径调用 get_history_imgs
            img_paths = get_history_imgs(image_full_path) 
            imgs = [Image.open(item) for item in img_paths]
            r_image = yolo.detect_image(imgs, crop = crop, count=count)
            list_img.append(cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')# *'XVID'           视频编解码器
        outfile = cv2.VideoWriter("output_video_path", fourcc, 5, (512, 512), True)    #大小必须和图片大小一致,且所有图片大小必须一致   -- photo_resize.py      
        # outfile = cv2.VideoWriter("./output.avi", fourcc, 5, (720, 480), True) 
        for i in list_img: 
            outfile.write(i) # 视频文件写入一帧
            #cv2.imshow('frame', next(img_iter)) 
            if cv2.waitKey(1) == 27: # 按下Esc键，程序退出(Esc的ASCII值是27，即0001  1011)
                break 
        outfile.release()
        cv2.destroyAllWindows()

    
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video'.")