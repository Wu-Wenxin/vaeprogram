项目重整理

```
vae-wwx
├── README.md
├── pyproject.toml    项目配置文件
├── configs/          配置文件夹，用来存放`.ymal`
├── examples/         示例脚本文件夹
├── tests/            测试文件夹            
└── src/              源码
```

读取配置 → 加载数据 → 创建模型 → 训练 → 保存结果

yaml 文件
  ↓
loader.py 读取
  ↓
普通 dict
  ↓
schema.py 规定格式
  ↓
ExperimentConfig 对象
  ↓
后面交给 model / data / train 使用

```code
x
↓
encoder
↓
feature
↓
latent head
↓
mu, logvar
↓
reparameterize
↓
z
↓
decoder
↓
x_hat
```