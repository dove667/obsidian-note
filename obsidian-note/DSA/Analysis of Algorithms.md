**Analysis of Algorithms** is a fundamental aspect of computer science that involves evaluating performance of algorithms and programs. Efficiency is measured in terms of **time** and **space**.

  

## **Asymptotic Analysis 渐进分析**

**给定一个任务的两种算法，我们如何找出哪一种更好？**

一个简单粗暴的方法是——实现这两种算法，然后在计算机上针对不同的输入运行这两个程序，看看哪一个花费的时间更少。这种方法在算法分析方面存在很多问题。

- **对于某些输入，**第一个算法可能比第二个算法表现更好。而对于某些**输入**第二个算法表现更好。
- 还可能的是，对于某些输入，第一个算法在一台**机器**上表现更好，而对于其他一些输入，第二个算法在另一台**机器**上表现更好。

> 渐近分析是解决算法分析中上述问题的重要思想。在渐近分析中，我们根据**输入大小**来评估算法的性能（我们不测量实际运行时间）。我们计算算法所花费的时间（或空间）随输入大小的增长顺序。

缺点：忽略了常数（比如机器能力），输入不一定充分大

  

### Worst, Average, and Best cases of an algorithm

|   |   |   |
|---|---|---|
|**Notation**|**Definition**|**Explanation**|
|Big O (O)|f(n) ≤ C * g(n) for all n ≥ n0，say f(n) = O(g(n))|Describes the upper bound of the algorithm’s running time in the **worst case**.|
|Ω (Omega)|f(n) ≥ C * g(n) for all n ≥ n0，say f(n) = Ω(g(n))|Describes the lower bound of the algorithm’s running time in the **best case**.|
|θ (Theta)|C1 * g(n) ≤ f(n) ≤ C2 * g(n) for n ≥ n0，say f(n) =θ(g(n))|Describes both the upper and lower bounds of the algorithm’s **running time**.|

_**Big-O notation** is used to describe the performance or complexity of an algorithm. Specifically, it describes the **worst-case scenario** in terms of **time** or **space complexity.**_  
微积分中的上界O,f最大和g同阶。O(g)表示复杂度f最大和g同阶。  

同理，Ω(g(n))表示最小和g同阶，θ(g(n))表示就是和g同阶。

### Properties

If f(n) = O(g(n)) and h(n) = O(k(n)), then f(n) * h(n) = O(g(n) * k(n)).

If f(n) = O(g(n)) and g(n) = O(h(n)), then f(g(n)) = O(h(n))

  

|   |   |   |
|---|---|---|
|**Type**|**Notation**|**Example Algorithms**|
|Logarithmic|O(log n)|Binary Search|
|Linear|O(n)|Linear Search|
|Superlinear|O(n log n)|Heap Sort, Merge Sort|
|Polynomial|O(n^c)|Strassen’s Matrix Multiplication, Bubble Sort, Selection Sort, Insertion Sort, Bucket Sort|
|Exponential|O(c^n)|Tower of Hanoi|
|Factorial|O(n!)|Determinant Expansion by Minors, Brute force Search algorithm for Traveling Salesman Problem|