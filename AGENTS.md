- 执行任务时，如果需要请求下载数据集/模型文件/请求大模型api服务，**请关闭代理，不要走代理流量**。如果会导致任务中断，请停止任务进行并提醒我。

- 创建conda虚拟环境或者conda、pip安装虚拟环境所需包时，**请关闭代理，不要走代理流量**。

- 禁止联网请求huggingface、modelscope下载模型到本地缓存文件夹或检查模型是否为最新版本，本项目用到的模型都已经下载到了本地。

- 当前用到的molmo模型本地目录为：/home/yy/.cache/huggingface/hub/models--allenai--Molmo-7B-D-0924/

- 当前用到的Qwen3-14B模型本地目录为：/home/yy/data/code/T-FunS3D/models/Qwen3-14B

- 当前改造实验用到的Qwen3-VL-8B模型本地目录为：/home/yy/data/code/T-FunS3D/models/Qwen3-VL-8B-Instruct
