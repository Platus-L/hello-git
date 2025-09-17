### Part 1.监督学习

> ​                                                             （x<sup>(i)</sup>, y<sup>(i)</sup>）训练 &rarr;  训练集 &rarr;  学习算法  &rarr; 假设h

$$
\begin{cases}
回归  \ \ \ &regression  \  \ \ \ \ &连续问题\\
分类  \ \ \ &classification  \  \ \ \ \ &离散问题

\end{cases}
$$

#### 1. 线性回归

​	假设  	
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$
​		其中 $\theta$ : 参数（权重w） $x_0=1$ ( $\theta_0$是截距 )

​	**代价函数 **
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^{n}(h_\theta(x)^{(i)} -y^{(i)})
$$


##### 	1 .1最小均方算法 LMS

​		我们需要***J(&theta;)***函数尽可能小 &rarr;  **批量梯度下降** ***batch gradient descent***
$$
\theta_j := \theta_j-\alpha\frac{\partial }{\partial \theta_j}J(\theta)
$$
​	

​		求偏导：
$$
\begin{align}
\frac{\partial}{\partial\theta_j}J(\theta) &= [\frac{1}{2}(h_\theta-y)^2]
\\&=(h_\theta-y)\frac{\partial}{\partial\theta_j}(h_\theta-y)
\\&=(h_\theta-y)x_j
\end{align}
$$
​		回代得到：
$$
\theta_j := \theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
​		故：
$$
\theta := \theta+\alpha\sum_{i=1}^{n}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}
$$
​      	**随机梯度下降**

​		循环{

​			*i=1* 到 *n* {
$$
\theta_j := \theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
​			}

​		}

​		故：
$$
\theta : = \theta + \alpha(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}
$$
​		

​		随机梯度下降会重复遍历训练集，每次遇到一个训练样本时，仅针对该单个样本计算误差梯度并更新参数。

​		当训练规模*n*很大时，批量梯度下降在执行单次更新前需要扫描整个训练集，这是一项昂贵的操作，而随机梯度下降可以立即开		始并对每个样本都取得进展。

​		因此在训练集很大时，通常倾向使用随机梯度下降

​	

### 1.2正规方程 

> Tips 

$$
\nabla x^TAx = 2Ax
$$

$$
h_\theta(x^{(i)}) = (x^{(i)})^T\theta
$$

​										则
$$
x\theta-\vec{y} = 
\begin{bmatrix}
(x^{(1)})^T\theta-y^{(1)}\\
\vdots \\
(x^{(n)})^T\theta-y^{(n)}
\end{bmatrix}
$$

$$
\begin{align}
\therefore J(\theta) &=\frac{1}{2}(x\theta-\vec y)^T(x\theta-\vec y) \\
\therefore \nabla_\theta J(\theta) 
&=\frac{1}{2}\nabla_\theta(\theta^Tx^Tx\theta-\vec{y}^Tx\theta-\theta^Tx^T\vec y+\vec{y}^T\vec{y})\\
& = \frac{1}{2}\nabla_\theta(\theta^Tx^Tx\theta-2\theta^Tx^T\vec y+\vec{y}^T\vec{y})\\
& = \frac{1}{2}(2x^Tx\theta-2\vec{y}^Tx)\\
& =x^Tx\theta-x^T\vec{y}
\end{align}
$$


$$
\therefore x^Tx\theta=x^T\vec{y} \ \  时 \ \nabla_\theta J(\theta)=0\\
\therefore \theta = (x^Tx)^{-1}x^T\vec{y}
$$


### 1.3概率

> 立个flag（：因为需要补数学知识QAQ

### 1.4局部加权线性回归

![1](F:\my_pytorch\fit_1.jpg)



​	上面最左边的图里，我们显示了用$y=\theta+\theta_1x$拟合到数据集，我们可能会觉得不够理想，也就是***欠拟合(underfitting)***；

​	于是，我们添加额外的特征***x<sup>2</sup>***，然后拟合模型$y = \theta_0+\theta_1x+\theta_2x^2$, 那么对数据的拟合效果可能会改善；

​	然而，过度添加特征也存在***过拟合(overfitting)***风险

​	所以，**特征的选择**决定了学习算法的优劣， 这里我们简介一下局部加权线性回归算法

​	局部加权线性回归是一种**非参数(non-parametric)**算法

​		1. 拟合&theta;以最小化
$$
\sum_{i}w^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2
$$
​		2.输出 $\theta^Tx$

​		一般的一种常见权重选择方法是
$$
w^{(i)} = exp(-\frac{(x^{(i)}-x)^2}{2\tau^2}) ^{*}
$$
​		权重$w$： 取决与评估的特定点$x$。

​			注意到，当$|x^{(i)}-x|$值很小时，$w(i)$接近于1；如果$|x^{(i)}-x|$很大时，$w(i)$会很小。

​			所以，靠近查询点x的训练样本会被赋予更高的权重。（注意，w与高斯分布无直接联系）

​		带宽参数*($bandwidth$)*$\tau$： 控制训练样本的权重随其与查询点x距离衰减速度

​		

__________________________________________

$^*$如果$x$是向量，则推广为$w(i)=exp(-\frac{(x^{(i)}-x)^T(x^{(i)}-x)}{2\tau^2})$或者$w(i)=exp(-\frac{(x^{(i)}-x)^T\Sigma^{-1}(x^{(i)}-x)}{2\tau^2})$其中$\tau和\Sigma$选择合适值













### Part 2.分类与逻辑回归

​	分类问题预测的目标变量y只能取有限的离散值。 

​	二元分类$(binary\ classifasion)$问题: $y$(**标签**)的取值仅限于$0$和$1$

#### 2.1 逻辑回归

​	**逻辑函数（$logistic\ function$）** 或**$S$形函数（$sigmoid\ function$）**。
$$
h_\theta(x) = g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}\\
其中g(z)=\frac{1}{1+e^{-z}}
$$
![sig](F:\my_pytorch\sigmoid.png)

​	令$x_0 = 1$，从而$\theta^Tx = \theta_0+\Sigma_{j=1}^{d}\theta_jx_j$

​	将$g$视为已知条件，特别的，记$g'$为$Sigmoid$函数的导数，则有
$$
\begin{aligned}
g'(z) &= \frac{d}{dz}\frac{1}{1+e^{-z}}\\
&=\frac{1}{(1+e^{-z})^2}e^{-z}\\
&=\frac{1}{(1+e^{-z})}(1-\frac{1}{(1+e^{-z})})\\
&=g(z)(1-g(z))
\end{aligned}
$$
​	概率假设，设
$$
\begin{cases}
P(y=1|x;\theta) &= h_\theta(x)\\
P(y=0|x;\theta) &= 1-h_\theta(x)
\end{cases}\\
\implies
p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$
​	假设n个训练样本相互独立，那么
$$
\begin{aligned}
L(\theta)&=p(\vec{y}|X;\theta)\\
&=\prod_{i=1}^{n}p(y^{(i)}|x^{(i)};\theta)\\
&=\prod_{i=1}^{n}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x))^{1-y^{(i)}}
\end{aligned}
$$

$$
\xrightarrow{取对数}l(\theta) = \Sigma_{i=1}^{n}y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))
$$

​	最大化 $l(\theta)$ 函数 &rarr; 梯度上升法 $\theta := \theta+\alpha\nabla_\theta l(\theta)$

​	我们从一个样本$(x,y)$出发， 推到随机梯度上升规则的导数
$$
\begin{aligned}
\frac{\partial}{\part\theta_j}l(\theta) &= (y\frac{1}{g(\theta^{T}x)}-(1-y)\frac{1}{1-g(\theta^Tx)})g(\theta^Tx)(1-g(\theta^Tx))\frac{\part}{\part\theta_j}\theta^Tx\\
&=(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx))x_j\\
&=(y-g(\theta^Tx))x_j\\
\\\\
\theta := \theta+\alpha\nabla_\theta &l(\theta)\xrightarrow{代入}\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
\end{aligned}
$$




> flag：逻辑回归更新规则与最小均方的在形式上相同：这个GLM时会给出解释
>
> ​	  而且，上面的$h_\theta(x^{(i)})$时非线性函数哦







