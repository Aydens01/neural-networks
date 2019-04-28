# Formal neuron

## Theory

To do...

## Unitary tests

To test the sigmoid function as the activation function for a formal neuron, we use these parameters : **[w1, w2] = [1, -1]** and **w0=10**. Moreover we will put the **a** parameter to **1000**. So we have three cases :
* x1 - x2 + 10 > 0 => y = 1.0
* x1 - x2 + 10 = 0 => y = 0.5
* x1 - x2 + 10 < 0 => y = 0.0
We can see it in the following chart

![Output chart](img\sigmoid.png)

*Note :* *In <span style="color:blue;">blue</span> we have the formal neuron output equals to 0 and in <span style="color:red;">red</span> to 1.*