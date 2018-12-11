# 贝叶斯决策

## 决策规则

1. 最小错误率准则
2. 最小风险准则
3. Neyman-Pearson准则
4. 最小最大决策准则

- ### 最小错误率准则

  以后验概率为判决函数：$ P\left ( \omega _{i} |\mathbf{x}\right ) $

  决策规则：$ j=\underset{ i }{\arg \max}P\left ( \omega _{i} |\mathbf{x}\right ) $

  等价于 $ \begin{split} 
  \underset{ i }{\arg \max}P\left ( \omega _{i} |\mathbf{x}\right )   = & \underset{ i }{\arg \max} \frac{p\left ( \mathbf{x}|\omega _{i} \right ) P\left ( \omega _{i} \right )}{p\left ( \mathbf{x} \right )}\\
  & =\underset{ i }{\arg \max} \ p\left ( \mathbf{x}|\omega _{i} \right ) P\left ( \omega _{i} \right )
  \end{split} $

  > 选择 $ P\left ( \omega _{1} |\mathbf{x}\right ) $，$ P\left ( \omega _{2} |\mathbf{x}\right ) $ 中最大值对应的类别j最为决策结果

  ##### 最小错误率准则二分类下的表述形式：

  1. 如果$ P\left ( \omega _{1} |\mathbf{x}\right ) $ > $ P\left ( \omega _{2} |\mathbf{x}\right ) $ ，则$x\in w_{1}$

     如果$ P\left ( \omega _{1} |\mathbf{x}\right ) $ < $ P\left ( \omega _{2} |\mathbf{x}\right ) $ ，则$x\in w_{2}$ 

  2. 如果 $ p\left ( \mathbf{x}|\omega _{1} \right ) P\left ( \omega _{1} \right )$ > $p\left ( \mathbf{x}|\omega _{2} \right ) P\left ( \omega _{2} \right )$ ，则 $x\in w_{1}$

  3. 如果 $\frac{p\left ( \mathbf{x}|\omega _{1} \right ) }{p\left ( \mathbf{x}|\omega _{2} \right ) }$ > $\frac{P\left ( \omega _{2} \right )}{P\left ( \omega _{1} \right )}$，则$x\in w_{1}$

  4. 如果$h\left ( x \right )=-\ln p\left ( x|\omega _{1} \right )+\ln p\left ( x|\omega _{2} \right ) < \ln \frac{P\left ( \omega _{1} \right )}{P\left ( \omega _{2} \right )}$，则$x\in w_{1}$

------

- ### 最小风险准则

- ### Neyman-Pearson准则

- ### 最小最大决策准则


