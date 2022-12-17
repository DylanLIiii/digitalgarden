---
{"dg-publish":true,"permalink":"/personal-page/test/"}
---

# Interest Rates and Fixed Income Instruments
### 利率收益
- 单利
- 复利
- 连续复利
### 现值(Present value)


Present Value (PV) assuming interest rate $r$ per period
$$
P V( c ; r)=c_{0}+\frac{c_{1}}{(1+r)}+\frac{c_{2}}{(1+r)^{2}}+\ldots \frac{c_{N}}{(1+r)^{N}}=\sum_{k=0}^{N} \frac{c_{k}}{(1+r)^{k}}
$$

> 我们对合约价值做无套利假设,有且仅当合约价值等于现值时无套利实现(**这是借贷利率相同的情况**)
### Different Lending and Borrowing Rate
Can lend at rate $r_{L}$ and borrow rate at rate $r_{B}: r_{L} \leq r_{B}$
- Portfolio: buy contract, and borrow $\frac{c_{k}}{\left(1+r_{B}\right)^{k}}$ for $k$ years, $k=1, \ldots, N$
	- Cash flow in year $k: c_{k}-\frac{c_{k}}{\left(1+r_{B}\right)^{k}}\left(1+r_{B}\right)^{k}=0$ for $k \geq 1$
	- No-arbitrage: price $=p-c_{0}-\sum_{k=1}^{N} \frac{c_{k}}{\left(1+r_{B}\right)^{k}} \geq 0$
	- Lower bound on price $p \geq P V\left( c ; r_{B}\right)$
- Portfolio: sell contract, and lend $\frac{c_{k}}{\left(1+r_{L}\right)^{k}}$ for $k$ years, $k=1, \ldots, N$ Cash flow in year $k:-c_{k}+\frac{c_{k}}{\left(1+r_{L}\right)^{k}}\left(1+r_{L}\right)^{k}=0$ for $k \geq 1$
	- No-arbitrage: price $=-p+c_{0}+\sum_{k=1}^{N} \frac{c_{k}}{\left(1+r_{L}\right)^{k}} \geq 0$
	- Upper bound on price $p \leq P V\left( c ; r_{L}\right)$
	- Bounds on the price $P V\left( c ; r_{B}\right) \leq p \leq P V\left( c ; r_{L}\right)$
	  
	  > 满足无套利的情况下实现盈利
### Fix Income securities(固定收入工具)
 > 不是无风险的,还需要承担违约风险.通胀风险和市场风险
#### 永续(Perpetuity)

$\begin{aligned} c_{k}=A \text { for all } k \geq 1 & \\ p=\sum_{k=1}^{\infty} \frac{A}{(1+r)^{k}}=\frac{A}{r} \end{aligned}$
> 可以看出永续合约是没有交割日期的,合约在平仓前是永久持有的($k趋向\infty$)
#### 年金(Annuity)
$c_{k}=A$ for all $k=1, \ldots, n$
$$
\begin{array}{l}
\text { Annuity }=\text { Perpetuity }-\text { Perpetuity starting in year } n+1 \\
\text { Price } p=\frac{A}{r}-\frac{1}{(1+r)^{n}} \cdot \frac{A}{r}=\frac{A}{r}\left(1-\frac{1}{(1+r)^{n}}\right)
\end{array}
$$
### Bonds(债券)
- Face value $F$ : usually 100 or 1000
- Coupon rate $\alpha$ : pays $c=\alpha F / 2$ every six months
- Maturity $T:$ Date of the payment of the face value and the last coupon
- Price $P$
- Quality rating: S\&P Ratings $AAA , AA , BBB , BB , CCC , CC$
#### 价格的制定
$$P=\sum_{k=1}^{2 T} \frac{c}{(1+\lambda / 2)^{k}}+\frac{F}{(1+\lambda / 2)^{2 T}}$$

$d(0,2) = \frac{1}{(1+s_2)^2} = 0.8750736155679097$