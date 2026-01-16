# xiaotudui_target_detection
b站小土堆的目标检测代码实操。刚开始注重整体把握，细节慢慢完善。


# 单目标训练
这个任务需要三个py文件：dataset.py(included the data),model.py,train_one_target.py
直接运行train_one_target.py即可。

xiaotudui没有这部分内容，因此这里是我自己用AI写的
训练流程：和之前的pytorch教程一样
主要的内容是数据的处理。
先把数据变为位置+one hot
其中位置要归一化方便处理
然后因为每个图有好几种类别且每个类别可能有好几个检测
因此这里是最简单的情况：挑出一种类别，若有多个，仅取第一个。

注：train.py是一个不完善的代码。
