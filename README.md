# RedTide
## 任务说明
使用时间序列异常检测方法，对预测未来三天得到的水质要素进行检测，以判断赤潮是否发生
## 目录结构
````
.
├── data_preprocess # 数据预处理
│   ├── open.py # 查看xlsx或csv文件
│   ├── utils_detection.py # 处理真实水质要素数据
│   └── utils_predict.py # 处理预测得到的水质要素数据
├── detection # 异常检测
│   ├── data_provider # 数据输入
│   │   ├── data_factory.py
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── exp # 实验逻辑
│   │   ├── exp_anomaly_detection.py
│   │   ├── exp_basic.py
│   │   └── __init__.py
│   ├── layers # 核心层
│   │   ├── Conv_Blocks.py
│   │   ├── Embed.py
│   │   └── __init__.py
│   ├── models # 模型框架
│   │   ├── FiLM.py
│   │   ├── __init__.py
│   │   └── TimesNet.py
│   ├── run.py # main函数
│   ├── run.sh # 执行脚本
│   └── utils # 辅助工具
│       ├── ADFtest.py
│       ├── augmentation.py
│       ├── dtw_metric.py
│       ├── dtw.py
│       ├── __init__.py
│       ├── losses.py
│       ├── masking.py
│       ├── metrics.py
│       ├── print_args.py
│       ├── timefeatures.py
│       └── tools.py
└── README.md
````