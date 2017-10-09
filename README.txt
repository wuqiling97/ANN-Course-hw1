额外的修改：
solve_net.py  加入了计算一个epoch的loss, accuracy并返回的代码
run_mlp.py  加入了保存每轮迭代的loss, accuracy并绘图；计算训练费时的代码。这些代码在73行之后，在复现结果的时候可以注释掉

添加的文件：
plot.py  定义了绘图的函数以便run_mlp调用