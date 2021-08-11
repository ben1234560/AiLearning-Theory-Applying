# README

**7th/Top1%，提供答疑**

![1628602069041](assets/1628602069041.png)

**也能做到前5，但是没必要**

![1628602545539](assets/1628602545539.png)

[竞赛地址](https://www.biendata.xyz/competition/didi-eta/)

持续更新中...

### 1.解题思路

[预估到达时间解题思路.pdf](https://github.com/ben1234560/AiLearning-Theory-Applying/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4%E8%A7%A3%E9%A2%98%E6%80%9D%E8%B7%AF.pdf)

<img src="assets/1628668115968.png" width="700" align="middle" />



### 2. 数据说明

- 由于滴滴数据保密协议，博主也无法找到可开放数据及数据地址，故无法提供。
- 数据来自滴滴出行，英文（Data source: Didi Chuxing），数据出处：[https://gaia.didichuxing.com](https://gaia.didichuxing.com/)

### 3. 特征说明

![1628670345575](assets/1628670345575.png)

![1628670144983](assets/1628670144983.png)

- max_order_xt：head级别的特征，如同一sample_eta、distinct等
- max_170_link_sqe_for_order：link序列特征，如右格式：[link_id_1, link_id_3, link_id_20...]
- cross_data_dir：cross序列特征
- link_data_other_dir：link统计特征，如某link_id前6小时的均值、求和等
- head_data_dir：历史同星期的全天的统计特征
- win_order_data_dir：订单的滑窗特征，如当前订单时间点的前段时间的统计特征
- arrival_data_dir：历史到达路况状态的统计特征
- zsl_arrival_data_dir：同上，不同人进行构建
- arrival_sqe_data_dir：到达时刻的序列特征，提供给DCN的T模型进行蒸馏给S模型
- pre_arrival_sqe_dir：利用树模型预测的到达时刻特征
- zsl_link_data_dir：link统计特征，不同人构建



### 4. 模型说明

- [DCN模型](https://github.com/ben1234560/AiLearning-Theory-Applying/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953)
  - ![1628669063602](assets/1628669063602.png)
- [WDR模型](https://github.com/ben1234560/AiLearning-Theory-Applying/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/WD_128544)
  - ![1628669073291](assets/1628669073291.png)
- LGB模型
  - ![1628669152380](assets/1628669152380.png)



### 5. 推荐工具

- [智能钛Notebook-2.4.0-tf](https://console.cloud.tencent.com/tione/notebook/instance)
- [腾讯云服务器](https://console.cloud.tencent.com/cvm/instance/index)



### 6. 环境配置和所需依赖库

- scikit-learn
- tqdm
- pandarallel
- joblib
- lightgbm
- pandas
- numpy
- keras_radam
- tensorFlow-gpu=2.4.0 

### 7. 文件说明

- [DCN_12953](https://github.com/ben1234560/AiLearning-Theory-Applying/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953)
  - DCN模型，线上分数0.12953
  - dcn_model/[dcn_model.py](https://github.com/ben1234560/AiLearning-Theory-Applying/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953/dcn_model/dcn_model.py)：模型代码
  - dcn_model/[main.py](https://github.com/ben1234560/AiLearning-Theory-Applying/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953/dcn_model/main.py)：主函数，训练和预测
  - dcn_model/[process.py](https://github.com/ben1234560/AiLearning-Theory-Applying/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953/dcn_model/process.py)：特征预处理
  - dcn_model/[model_h5](https://github.com/ben1234560/AiLearning-Theory-Applying/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/DCN_12953/model_h5)：存放处理信息，不影响模型结果
- [WD_128544](https://github.com/ben1234560/AiLearning-Theory-Applying/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AB%9E%E8%B5%9B%E5%AE%9E%E6%88%98_%E4%BC%98%E8%83%9C%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/%E6%BB%B4%E6%BB%B4%E2%80%94%E2%80%94%E9%A2%84%E4%BC%B0%E5%88%B0%E8%BE%BE%E6%97%B6%E9%97%B4/WD_128544)
  - WD模型，线上分数0.128544
  - 其他同上

### 8. 其他说明

- 代码属于公司所有，不能提供最优代码
- 感谢[@xbder](https://github.com/xbder)、[@AiIsBetter](https://github.com/AiIsBetter)

