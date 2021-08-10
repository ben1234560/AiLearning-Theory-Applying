# README

19th/Top2%，提供答疑

![1628577778121](assets/1628577778121.png)

<https://www.kaggle.com/c/indoor-location-navigation>

### 推荐notebook工具

- kaggle自带的notebook
- [智能钛Notebook-2.4.0-tf](https://console.cloud.tencent.com/tione/notebook/instance)



### How to run the code

1. 下载数据到input文件夹
2. floor预测代码
     part1 数据预处理
     运行code/wifi-features.ipynb 
     运行code/create-unified-wifi-features-example.ipynb
     part2 深度学习模型
     运行code/floor-model-blstm.ipynb
3. 坐标预测代码
     part1 数据预处理
     运行code/wifi-label-encode.ipynb
     运行code/data_abstract_sensor.ipynb
     运行code/data_abstract_wifi.ipynb
     运行code/gen_accl.ipynb
     part2 深度学习模型
     运行code/lstm-wifi-encode-wifi.ipynb   仅使用wifi数据预测
     运行code/lstm-wifi-encode-wifi-sensor.ipynb   使用wifi+sensor数据预测
4. 结果融合
     运行code/combine_v1.ipynb  模型线性融合
5. 后处理
     运行code/post_process.ipynb
6. 规则预测代码
     运行code/rules_infer.ipynb
7. 结果融合
     运行code/combine_v2.ipynb  模型线性融合得到最终final.csv预测文件