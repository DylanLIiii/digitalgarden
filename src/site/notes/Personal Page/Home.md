---
{"dg-publish":true,"permalink":"/personal-page/home/","tags":"gardenEntry"}
---

Hello! My name is Heng Li. 

  这是我第一次一个人完成研究内容, 除了导师会给出一些指导性的方向, 大部分的内容都由我自己来, 小到文献的管理, 大到每一次实验结果的保存, 模型和方法轮的改进 (这是非常重要的部分, 我到目前为止进度为 50% 已经进行了 100 多次实验, 我需要对每一次实验都有明确的控制和标签, 非常考验我的个人能力, 并且是我第一次接触, 从一开始跌跌撞撞进行研究, 到一个半月之后渐入佳境, 我学习到了许多新技能. 
	- 加深了我对相关领域的理解, 明确了未来一段时间需要学习的方向. 第一次自己单独完成整个 Project 的内容, 我发现了许多之前未曾注意到的地方, 比如要对模型进行一些改进, 可能会更加注重算法的改进, 而并不是代码实现, 再比如我一度困扰于如何 Handle 这次科研中巨大的数据量, 我为此学习了很多代码的优化方法和许多工程上的优化技巧 (也需要牺牲一些精度 -- 你的结果和你的成本代价总是相反的, 很多时候我能够得到更好的结果, 但是迫于需要敏捷实验, 只能降低精度换取更快的速度). 经过这一次科研 (仍在继续), 我更加了解了整个机器学习 Pipeline 的构建, 并且学习到了许多知识, 为未来的学习打下了基础, 明确了我在这个领域需要精进的不只是简单入门的代码能力 (怎么写模型, 简单的数据处理等等), 更多的会有如何多平台部署 , 如何抽象一个复用组件, 如何管理实验, 如何从数学上改进模型, 如何进行代码速度优化 (实际上这些在去年实习也有注意到, 但是没去过多研究, 这次深入研究了)


--- 
> [! Info] Resume
> - Collected 1.2 billion option data and developed a data preprocessing pipeline on AWS to clean/filter and reduce the data size by 91.7%.
> - Designed adversarial validation via TabFPN to detect the data leakage and determine the degree of data imbalance in order to improve the robustness of the model.
> - Implemented a training pipeline via Weight & Bias to train 12 models including improved DNN with skip-layers, M-DNN, Xgboost, and AutoGluon with GhostBatchGradient on 8 clustering dataset processed by MiniBatchKMeans which can avoid Overestimate and automatically tune hyperparameters on 3 different sliding window with bayesian and pruning strategies while log and visualize all experimental results (more than 500 experiments) and model hyperparameters.