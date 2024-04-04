# My solutions of CS229 2018 : Machine Learning

[course link](https://www.youtube.com/watch?v=8NYoQiRANpg)

[repo link](https://github.com/maxim5/cs229-2018-autumn)

## Progress
- [x] Problem-Set-0
- [x] Problem-Set-1
- [ ] Problem-Set-2
- [ ] Problem-Set-3
- [ ] Problem-Set-4

## Info about Problem-Sets

### Problem-Set-0

主要是一些简单的微积分、线性代数和概率计算，就没有放上来了。

### Problem-Set-1

第一题研究了Logistic回归和GDA高斯判别分析，证明了Logistic回归的损失函数是凸的，这保证了梯度下降能收敛到全局最优解；然后
证明了课上的一个结论，在GDA中如果我们将对概率服从高斯分布的假设中的方差 $\Sigma$ 固定，那么GDA的决策边界是线性的；还计算了一下
各个参数的极大似然估计的结果(官方答案中采取了 $n=1$ 的假设，在仓库中丢掉了这个假设进行计算)。[该仓库中的最后一问并没有详细给出Box-Cox transformation的过程，以后有空补一下]

第二题研究了当模型为Logitic模型的时候，PU Learning的做法，本题比较简单，需要注意的是最后一问中correction的计算方法，目的是希望将训练得到的适用于 $y$ 标签的 $\theta$ 通过变换
得到适用于 $t$ 标签的 $\theta'$。

第三题研究了Possion回归的模型，证明了Possion是指数族中的一员。

第四题是研究指数族中的成员的损失函数的凸性，同时通过计算证明了 $\mathbb{E}[Y | X; \theta] = \frac{\partial}{\partial \eta} \alpha(\eta)$ 以及 
$\mathbf{Var}[Y | X; \theta] = \frac{\partial^2}{\partial \eta^2} \alpha(\eta)$。

第五题研究了在局部赋予权重的情况下的线性回归模型。

---

**Thanks to [Eiko](https://github.com/Eiko-Tokura) for helping me with these!**
