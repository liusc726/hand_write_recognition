# HandwriteRecognition
a handwritten text recognition system for Huawei atlas200dk 
配置环境请参考官网： https://www.huaweicloud.com/ascend/home.html

# 演示视频
可以实现自然场景中边写字边识别的功能

# 目录说明
## 1. HandwriteRecognition文件夹
配置完整的项目文件，配置模型为891版本，请在对应版本的DDK下运行，直接按照该文件下的 文字检测部署步骤.txt，部署运行即可
更详细的开发指导请参考开发指导书
## 2. 3755caffe文件夹
训练好的包含3755类汉字识别的caffe模型，可用于自己转化模型，转化过程请参考开发指导说明书最后模型转换
## 3. om_model文件夹
转换好的om模型，对应不同的ddk版本分为750和891版本，两者主要区别为在不同的DDK版本下使用模型转换工具转换而成，请依据自己安装的DDK版本选择。
#
## *部署详情请看 部署说明.doc*
#

