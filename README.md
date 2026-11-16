# 基于机器学习的糖尿病病情进展数值预测

使用 scikit-learn 的 `diabetes` 数据集，根据基线特征预测一年后的病情进展指标。数据包含 442 位患者，每位患者有 10 个数值特征，包括年龄、性别编码、BMI、平均血压和 6 项血清测量。实验使用 `scaled=False` 读取未缩放的特征值。[数据集说明](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

## 运行实验

在项目目录的 PowerShell 中执行。结果使用 Python 3.13.9 和 scikit-learn 1.9.1 复现。若已有虚拟环境 `.venv`，可从第二条命令开始。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

脚本输出 5 折训练与验证结果、最佳参数和测试集上的三项指标。如需生成真实值与预测值散点图，在项目虚拟环境中安装 Matplotlib，再运行：

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib==3.11.2
.\.venv\Scripts\python.exe main.py --plot
```

图表保存为 [predictions.png](predictions.png)。每个面板绘出同一批 89 位测试患者，横轴为真实值，纵轴为预测值。点在虚线上方表示高估，下方表示低估。四个面板共用坐标范围。

## 实验方法

1. 用 `random_state=42` 把 442 条记录划分为 353 条训练记录和 89 条测试记录，沿用基线实验的划分。
2. 用相同的 5 折划分比较两种基线。线性回归拟合特征的加权和，随机森林用 100 棵回归树的平均预测。
3. 在 353 条训练记录上做网格搜索，按 5 折平均验证 RMSE 选参数。随机森林固定 100 棵树，对最大深度 `[None, 3, 5, 8, 12]` 和叶节点最小样本数 `[1, 3, 5, 10]` 做组合搜索，包含原有配置。岭回归用 `Pipeline` 在每个训练折单独拟合 `StandardScaler`，再用 `Ridge` 约束系数大小。正则化强度 `alpha` 的候选值为 `[0.01, 0.1, 1, 10, 30, 100, 300]`。[网格搜索](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) · [预处理与数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) · [岭回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
4. 用完整训练集拟合确定的配置，在那 89 条测试记录上比较 MAE、RMSE 和 R²。MAE 是平均绝对误差，RMSE 对较大的误差更敏感，两者越小越好。R² 衡量相对于用真实目标均值作统一预测时，平方误差减少的比例，越大越好。[回归评价指标](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

## 交叉验证结果

下表的训练和验证 RMSE 均为 5 折的平均值。岭回归和调参随机森林的验证结果来自用于选择参数的同一组折。

| 模型 | 主要参数 | 训练 RMSE ↓ | 验证 RMSE ↓ |
|---|---|---:|---:|
| 线性回归 | 默认 | 53.35 | 55.39 |
| 随机森林 | `max_depth=None`，`min_samples_leaf=1` | 22.00 | 59.49 |
| 岭回归 | `alpha=1` | 53.37 | **55.37** |
| 调参随机森林 | `max_depth=12`，`min_samples_leaf=5` | 38.39 | 57.89 |

原始随机森林的训练与验证 RMSE 相差 37.49。限制树的复杂度后，差距缩小到 19.50，而且五折验证 RMSE 均下降。岭回归的平均验证 RMSE 比线性回归低约 0.02，在五折中有两折更低，两种线性模型的表现接近。

## 测试结果与结论

| 模型 | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| 线性回归 | **42.79** | 53.85 | 0.453 |
| 随机森林 | 44.08 | 54.35 | 0.443 |
| 岭回归 | 42.81 | 53.78 | 0.454 |
| 调参随机森林 | 43.54 | **53.60** | **0.458** |

限制树的复杂度把随机森林的测试 RMSE 从 54.35 降到 53.60，降幅为 0.75。它在原测试划分中取得最低 RMSE 和最高 R²，线性回归取得最低 MAE。按预先选定的平均验证 RMSE，岭回归略领先。测试 RMSE 最低的三个模型相差不到 0.3，排名仅反映这次样本划分和候选参数范围。原测试集的基线分数在调参前已查看，网格搜索仅使用训练集上的交叉验证成绩。

散点图中，四种模型都低估了真实值接近 300 的几位测试患者。整体 RMSE 接近时，仍可以通过对照每个样本与虚线的距离，观察误差集中在哪些真实值范围。

实验实现参考了 [Scientific Agent Skills](https://arxiv.org/abs/2609.00065) 中的 scikit-learn 评估与科学可视化流程。
