# 历史成绩说明

|项目|记录|
|---|---|
|比赛/榜单|2026跨本体具身认知极限联合挑战赛，初赛A|
|历史版本/成绩|v4，截图最佳成绩17.1028439153439；提交记录显示17.1028|
|正式提交时间|2026-09-25 18:08:45，Asia/Shanghai|
|策略身份|`rgb3-qwen-closed-loop` / `v4`|
|历史模型|`qwen3.8-omni-flash`，外部API，temperature=0|
|配置|同一run-id，30题×2本体，每格7200步，H=X=1|
|官网成绩截图|[原始截图](assets/score.png)，作者提供，未修改像素或元数据|
|公开发布|[ccic-v4-17.1028 Release](https://github.com/ben1234560/AiLearning-Theory-Applying/releases/tag/ccic-v4-17.1028)，2026-10-18发布；不是新的线上评测|

历史提交文件：[baseline-v4-OFFICIAL_30X2.zip](https://github.com/ben1234560/AiLearning-Theory-Applying/releases/download/ccic-v4-17.1028/baseline-v4-OFFICIAL_30X2.zip)，2,059,396,429字节（约1.92GiB）。SHA256：

```text
1c592683ce15aa327e66a7347d381c1cd03a6580282a09fc38f0585d52cf28fd
```

可在下载后执行`shasum -a 256 baseline-v4-OFFICIAL_30X2.zip`，Linux也可用`sha256sum`；应与上面的值一致。2026-10-18核对的GitHub资产摘要与该历史哈希相同。

60/60格技术有效不代表60题成功；截图来自历史提交，不保证重新运行同分。分享版通过CPU接口检查，未进行完整GPU复跑或Docker验收。

`code/policy.py`与得分策略完全相同，哈希和包装改动见[provenance.json](configs/provenance.json)。默认改为单worker，移除凭据文件入口，修正GPU启动hook换行；未引入新控制算法，包装调整未经GPU重跑。
