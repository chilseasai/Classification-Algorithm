import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

import scala.util.Random

/**
 * Created by tseg on 2015/11/18.
 */
class SVMWithRBF(var input: RDD[LabeledPoint], var C: Double, var eps: Double, var tolerance: Double, var gamma: Double) extends Serializable {

  val N: Int = input.count().toInt // 样本数总数
  val trainNum: Int = N // 训练数据数目
  val d: Int = input.map(_.features.size).first() // 样本维数
  var target = new Array[Int](N) // 训练与测试样本的目标值
  var b = 0.0 // 阀值
  var errorCache = new Array[Double](trainNum) // 存放non-bound样本误差,拉格朗日乘子既非0也非C(non-bound，注意non-bound是针对Ci≤≤α0这一约束条件而言的)

  println("维数：-----------------： " + d)

  val data = input.map(lp => (lp.label, lp.features))
  data.cache()

  val dataMatrix = data.map { case (label, features) => // 列存储方式的训练样本矩阵(column-major)
    new DoubleMatrix(features.size, 1, features.toArray:_*)
  }.collect()

  val label = data.map { case (label, features) => // 标签数组
    label
  }.collect()

  val labelMatrix = new DoubleMatrix(label.length, 1, label:_*) // 标签列矩阵

  val alpha = DoubleMatrix.zeros(trainNum) // 创建一个大小为训练样本个数的全零列向量（矩阵）

  val precomputedSelfDotProduct = data.map { case (label, features) => // 预存自己的点积
    val weights = new DoubleMatrix(features.size, 1, features.toArray:_*)
    weights.dot(weights)
  }.collect()


//  /**
//   * 计算点积函数
//   * @param i1 第i1个数据点
//   * @param i2 第i2个数据点
//   * @return
//   */
//  def dotProductFunc(i1: Int, i2: Int): Double = {
//    val dot = dataMatrix(i1).dot(dataMatrix(i2))
//    dot
//  }


  /**
   * 径向基核函数RBF
   * @param i1 第i1个数据点
   * @param i2 第i2个数据点
   * @return
   */
  def kernelFunc(i1: Int, i2: Int): Double = {
    var s = dataMatrix(i1).dot(dataMatrix(i2))
    s *= -2
    s += precomputedSelfDotProduct(i1) + precomputedSelfDotProduct(i2)
    math.exp(-s * gamma)
  }


  /**
   * 径向基核函数RBF2
   * @param i1 第i1个训练样本点
   * @param point 测试样本点
   * @return
   */
  def kernelFunc(i1: Int, point: DoubleMatrix): Double = {
    var s = dataMatrix(i1).dot(point)
    s *= -2
    s += precomputedSelfDotProduct(i1) + point.dot(point)
    math.exp(-s * gamma)
  }


  /** 分类决策函数
    * 评价分类学习函数 f(j) = sumi(ai * yi * Kij) + b
    * @param k 对第k个点，判断其标签类型
    * @return
    */
  def learnedFun(k: Int): Double = {
//    ------思路：稀疏向量内积计算分类决策函数，这样做必须求trainNum次kernelFunc，慢！
    var s: Double = 0
    for(i <- 0 until trainNum) {
      if(alpha.get(i) > 0)
        s += alpha.get(i) * label(i) * kernelFunc(i, k)
    }
    s -= b
    s
  }


  /**
   * 评价分类学习函数2 f(j) = sumi(ai * yi * Kij) + b
   * @param k 对点k进行类别判断
   * @return
   */
  def learnedFun2(k: DoubleMatrix): Double = {
    var s = 0.0
    for(i <- 0 until trainNum) {
      if(alpha.get(i) > 0)
        s += alpha.get(i) * label(i) * kernelFunc(i, k)
    }
    s -= b
    s
  }


  /**
   * 优化两个乘子，成功返回 1，失败返回 0
   * @param i1 优化第i1个乘子
   * @param i2 优化第i2个乘子
   * @return
   */
  def takeStep(i1: Int, i2: Int): Int = {
    if(i1 == i2) // 不会优化两个同一样本
      return 0

    val alpha1: Double = alpha.get(i1) // 记录乘子alphai1的旧值
    val alpha2: Double = alpha.get(i2) // 记录乘子alphai2的旧值
    var a1: Double = 0.0 // 记录乘子alphai1的新值
    var a2: Double = 0.0 // 记录乘子alphai2的新值
    val y1: Double = label(i1)
    val y2: Double = label(i2)

    var E1: Double = 0.0
    var E2: Double = 0.0
    if(alpha1 > 0 && alpha1 < C)
      E1 = errorCache(i1)
    else
      E1 = learnedFun(i1) - y1 // learnedFun(Int)为非线性的评价函数，即输出函数
    if(alpha2 > 0 && alpha2 < C)
      E2 = errorCache(i2)
    else
      E2 = learnedFun(i2) - y2

    val s = y1 * y2 // s指示y1和y2是否相等，或者说是否有相同的符号
    // 计算乘子的上下限
    var L = 0.0
    var H = 0.0
    if(s == 1) {  // y1和y2符号相同
      L = math.max(0, alpha1 + alpha2 - C)
      H = math.min(C, alpha1 + alpha2)
    }
    else {
      L = math.max(0, alpha2 - alpha1)
      H = math.min(C, C + alpha2 - alpha1)
    }

    if(L == H) return 0

    // 计算eta
    val k11 = kernelFunc(i1, i1)
    val k22 = kernelFunc(i2, i2)
    val k12 = kernelFunc(i1, i2)
    val eta = k11 + k22 - 2 * k12

    if(eta < -0.001) {
      val c = y2 * (E2 - E1) //书里写的是：c = y2 * (E1 - E2);
      a2 = alpha2 + c / eta // 计算未经剪辑时alpha2的值

      // 调整 a2 ，使其处于可行域，也就是对alpha2进行剪辑
      if(a2 < L)
        a2 = L
      else if(a2 > H)
        a2 = H
    }
    else { //分别从端点H,L求目标函数值Lobj,Hobj，然后设a2为所求得最大目标函数值
    val c1 = eta / 2
      val c2 = y2 * (E1 - E2) - eta * alpha2
      val Lobj = c1 * L * L + c2 * H
      val Hobj = c1 * H * H + c2 * H
      if(Lobj > Hobj + eps)
        a2 = L
      else if(Hobj > Lobj + eps)
        a2 = H
      else
        a2 = alpha2
    }

    //    论文写法：
    //    if (a2 < 1e-8)
    //      a2 = 0
    //    else if (a2 > C-1e-8)
    //      a2 = C
    //    if (|a2-alph2| < eps*(a2+alph2+eps))
    //      return 0
    //    a1 = alph1+s*(alph2-a2)

    if(math.abs(a2 - alpha2) < eps) //论文写法：if(math.abs(a2 - alpha2) < eps*(a2 + alpha2 + eps))
      return 0

    a1 = alpha1 + y1 * y2 * (alpha2 - a2) // 计算新的a1
    if(a1 < 0) { // 调整a1，使其符合条件
      a2 += s * a1
      a1 = 0
    } else if(a1 > C) {
      a2 += s * (a1 - C)
      a1 = C
    }

    // 更新罚值b
    var b1 = 0.0
    var b2 = 0.0
    var bnew = 0.0
    //    书上写法
    //    double b1 = b - Ei - y[i] * (a[i] - oldAi) * k(i, i) - y[j] * (a[j] - oldAj) * k(i, j);
    //    double b2 = b - Ej - y[i] * (a[i] - oldAi) * k(i, j) - y[j] * (a[j] - oldAj) * k(j, j);
    if(a1 > 0 && a1 < C) {
      bnew = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12
    } else {
      if(a2 > 0 && a2 < C)
        bnew = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22
      else {
        b1 = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12
        b2 = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22
        bnew = (b1 + b2) / 2
      }
    }
    val deltaB = bnew - b
    b = bnew

    /*
    对于线性情况，要更新权向量，这里不用了    
    更新error_cache，对于更新后的a1,a2,对应的位置i1,i2的error_cache[i1] =  error_cache[i2] = 0
    */
    val t1 = y1 * (a1 - alpha1)
    val t2 = y2 * (a2 - alpha2)
    for(i <- 0 until trainNum) {
      if(alpha.get(i) > 0 && alpha.get(i) < C)
        errorCache(i) += t1 * kernelFunc(i1, i) + t2 * kernelFunc(i2, i) - deltaB
    }
    errorCache(i1) = 0
    errorCache(i2) = 0
    alpha.put(i1, a1)
    alpha.put(i2, a2)
    1
  }


  /**
   * 1：在non-bound 乘子中寻找maximum fabs(E1-E2)的样本
   * @param i1
   * @param E1
   * @return
   */
  def examineFirstChoice(i1: Int, E1: Double): Int = {
    var i2: Int = -1
    var tmax: Double = 0.0
    for(k <- 0 until trainNum) {
      if(alpha.get(k) > 0 && alpha.get(k) < C) {
        val E2 = errorCache(k)
        if(math.abs(E1 - E2) > tmax) {
          i2 = k
          tmax = math.abs(E1 - E2)
        }
      }
    }
    if(i2 >= 0 && takeStep(i1, i2) == 1)
      return 1
    0
  }


  /**
   * 2：如果上面没取得进展，那么从随机位置查找non-boundary样本
   * @param i1
   * @return
   */
  def examineNonBound(i1: Int): Int = {
    val k0 = Random.nextInt(trainNum)
    for(k <- 0 until trainNum) {
      val i2 = (k0 + k) % trainNum
      if(alpha.get(i2) > 0 && alpha.get(i2) < C && takeStep(i1, i2) == 1)
        return 1
    }
    0
  }


  /**
   * 3：如果上面也失败，则从随机位置查找整个样本，改为bound样本
   * @param i1
   * @return
   */
  def examineBound(i1: Int): Int = {
    val k0 = Random.nextInt(trainNum)
    for(k <- 0 until trainNum) {
      val i2 = (k0 + k) % trainNum
      if(takeStep(i1, i2) == 1)
        return 1
    }
    0
  }


  /**
   * 假定第一个乘子ai（位置为 i1），examineExample(i1)首先检查，如果它超出 tolerance 而违背KKT条件，
   * 那么它就成为第一个乘子；
   * 然后，寻找第二个乘子（位置为 i2），通过调用 takeStep(i1, i2)来优化这两个乘子
   */
  def examineExample(i1: Int): Int = {
    val y1 = label(i1)
    val alpha1 = alpha.get(i1)
    val E1 = if(alpha1 > 0 && alpha1 < C) errorCache(i1) else learnedFun(i1) - y1
    val r1 = y1 * E1
    if((r1 > tolerance && alpha1 > 0) || (r1 < -tolerance && alpha1 < C)) {
      /**
       * 使用三种方法选择第二个乘子
       * 1：在non-bound乘子中寻找maximum fabs(E1-E2)的样本
       * 2：如果上面没有取得进展，那么从随机位置查找non-bound样本
       * 3：如果上面也失败，则从随机位置查找整个样本，改为bound样本
       */
      if(examineFirstChoice(i1, E1) == 1) // 第一种情况
        return 1
      if(examineNonBound(i1) == 1) // 第二种情况
        return 1
      if(examineBound(i1) == 1) //第三种情况
        return 1
    }
    return 0
  }


  /**
   * 计算误差率
   */
  def errorRate(data: RDD[(Int, DoubleMatrix)]): Double = {
    data.cache()
    val n = data.count().toDouble
    var ac = 0
    println("----------------- 测试结果 -----------------")
    val accuracy = data.map { case (label, features) =>
      val tar = learnedFun2(features)
      if ((tar > 0 && label > 0) || (tar < 0 && label < 0))
//      if ((tar > 0 && label > 0) || (math.abs(tar) < 1.0E-8 && label == 0))
        ac = 1
      ac
    }.reduce(_+_) / n

    println("准确率：" + accuracy * 100 + "%")
    accuracy
  }


  def run(): SVMModel = {
    /**  
      * 以下两层循环，开始时检查所有样本，选择不符合 KKT 条件的两个乘子进行优化，选择成功，返回 1，否则，返回 0  
       * 所以成功了，numChanged 必然大于 0，从第二遍循环时，就不从整个样本中去寻找不符合 KKT 条件的两个乘子进行优化， 
       * 而是从边界的乘子中去寻找，因为非边界样本需要调整的可能性更大，边界样本往往不被调整而始终停留在边界上。  
      * 如果，没找到，再从整个样本中去找，直到整个样本中再也找不到需要改变的乘子为止，此时，算法结束。  
      */
    var numChanged = 0  // number of alpha[i], alpha[j] pairs changed in a single step in the outer loop  
    var examineAll = 1  // flag indicating whether the outer loop has to be made on all the alpha[i]
    while(numChanged > 0 || examineAll == 1) {
      numChanged = 0
      if(examineAll == 1) {
        for(k <- 0 until trainNum)
          numChanged += examineExample(k) // 检查所有样本
      }
      else {
        for(k <- 0 until trainNum) {
          if(alpha.get(k) != 0 && alpha.get(k) != C)
            numChanged += examineExample(k) // 寻找所有非边界样本的lagrange乘子
        }
      }
      if(examineAll == 1)
        examineAll = 0
      else if(numChanged == 0) // else if(!numChanged)
        examineAll = 1
    }

    // 存放训练后的参数
    var supportVectors = 0
    for(i <- 0 until trainNum) {
      if(alpha.get(i) > 0 && alpha.get(i) <= C)
        supportVectors += 1
    }
    println("================ supportVectors 个数: ================ " + supportVectors)

    val test = data.map { case (label, features) => // 列存储方式的训练样本矩阵(column-major)
      val feature = new DoubleMatrix(features.size, 1, features.toArray:_*)
      (label.toInt, feature)
    }

    errorRate(test)

    new SVMModel()
  }


}
