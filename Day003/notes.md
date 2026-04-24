np.array()把里面的元素变成数据类型为  np.ndarray类型

copy是np.array(ob，dtype，copy=)的可选参数，默认值为 True：
 1.默认copy=True时，无论输入是列表 / 元组还是 ndarray，都会创建一个全新的数组副本，新数组和原数组相互独立，内存地址不同。

  2.仅当 ** 输入已经是 ndarray，且手动设置copy=False** 时，才不会创建新副本，直接返回原数组的内存引用（和原数组是同一个对象）。

  3.易错考点：如果输入是 Python 列表、元组等原生序列，哪怕设置copy=False，也会创建新的 ndarray 数组 —— 因为必须把原生数据转换为 ndarray 的内存结构，只有输入本身就是 ndarray 时，copy=False才会生效。
  np.asarray()	 把一个数据变成ndarray型
    仅当输入不是 ndarray 时，才创建**新数组**；输入为 ndarray 时，永远不创建副本，直接返回原引用
np.linspace(start, stop, num, endpoint=False)的 4 个核心参数：
 1.start=0：等差数列的起始值，生成的数组第一个元素就是 0
 2.stop=1：等差数列的终止参考值，最终数组是否包含这个值，由endpoint参数决定
 3.num=10：生成的数组固定的元素总个数，这是 PPT 里这句话的核心 ——linspace是通过「指定元素数量」来生成等差数列，元素之间的间隔会自动计算
 4.endpoint=False：关键开关参数，意思是最终生成的数组不包含终止值 1。补充：endpoint默认值为True，也就是默认包含终止值。
np.arange()：控制「元素之间的间距（步长）」，元素个数自动计算
 1.你只需要告诉它「起始值、终止值、两个数之间差多少」，它会自动算出最终有多少个元素。
 2.语法规则：和 Python 原生range一致，左闭右开，默认不包含终止值 stop。对应 PPT 里的示例：np.arange(0,5,0.5)起始值 0，终止值 5，步长（间距）0.5

属性.shape, .dtype, .ndim, .size的用法和示例。
.shape：返回一个元组，表示数组在每个维度上的大小，即数组的形状。
.dtype ：返回数组元素的数据类型。
.ndim ：返回数组的维数。
.size ：返回数组中的总元素数量。

