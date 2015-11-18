import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix


import scala.collection.mutable.ArrayBuffer
import scala.util.Random


/**
 * Created by tseg on 2015/10/30.
 */
class SVMTest(var input: RDD[LabeledPoint], var C: Double, var eps: Double, var tolerance: Double, var gamma: Double) extends Serializable {

  // 样本数总数
  val N: Int = input.count().toInt
  // 训练数据数目
  val trainNum: Int = N
  // 样本维数
  val d: Int = input.map(_.features.size).first()

  //  val twoSigmaSquared = 2.0 // RBF核函数中的参数
  // 训练与测试样本的目标值
  var target = new Array[Int](N)
  // 阀值
  var b = 0.0
  // 存放样本的矩阵
  var densePoints = Array.ofDim[Double](trainNum, d)// （old：存放训练与测试样本，0~train_num-1 训练；train_num~N-1 测试）
  // 拉格朗日乘子
  var alpha = new Array[Double](trainNum)
  // 存放non-bound样本误差,拉格朗日乘子既非0也非C(non-bound，注意non-bound是针对Ci≤≤α0这一约束条件而言的)
  var errorCache = new Array[Double](trainNum)
  // 预存dotProductFunc(i,i)的值，以减少计算量
  var precomputedSelfDotProduct = new Array[Double](N)

  println("维数：-----------------： " + d)

  val data = input.map(lp => (lp.label, lp.features))
//  data.foreach(arg => println(arg._2.apply(127)))

//  data.foreach(arg => println(arg._2))

//  val weights = data.map(lp => lp._2.toArray).first()
//  weights.foreach(v => print(v + " "))

//  densePoints = data.map { case (label, features) =>
//    var weights = new DoubleMatrix(features.size, 1, features.toArray:_*)
//    weights
//  }.collect()

//  target.foreach(println)
//
//  densePoints.foreach { row =>
//    row.foreach(column =>
//      print(column + " ")
//    )
//    println()
//  }



//    var i = 0
//    var matrixPoints = data.map { case (label, features) =>
//      target(i) = label.toInt
//      densePoints.getRow(i) = new DoubleMatrix(1, features.size, features:_*)
//      i += 1
//      densePoints
//    }.collect()

  ////////////////////测试用///////////////////////
  //  var i = 0
  //  val E1 = alpha.map(al =>
  //    if(al > 0 && al < C) errorCache(i + 1)
  //    else learnedFun(i + 1) - target(i + 1)
  //  )
  ////////////////////////////////////////////////
  /*
  /**
   * 计算点积函数
   * @param i1
   * @param i2
   * @return
   */
  def dotProductFunc(i1: Int, i2: Int) = {
    val dot = densePoints.getRow(i1).dot(densePoints.getRow(i2))
    dot
  }

  /**
   * 径向基核函数RBF
   * @param i1
   * @param i2
   * @return
   */
  def kernelFunc(i1: Int, i2:Int) = {
    var s = dotProductFunc(i1, i2)
    s *= -2
    s += precomputedSelfDotProduct(i1) + precomputedSelfDotProduct(i2)
    math.exp(-s * gamma) // math.exp(-s / twoSigmaSquared)
  }

  /**
   * 径向基核函数RBF2
   * @param i1
   * @param data
   * @return
   */
  def kernelFunc(i1: Int, data: DoubleMatrix) = {
//    var s = dotProductFunc(i1, i2)
    var s = densePoints.getRow(i1).dot(data)
    s *= -2
    s += densePoints.getRow(i1).dot(densePoints.getRow(i1)) + data.dot(data)
    math.exp(-s * gamma) // math.exp(-s / twoSigmaSquared)
  }

  /**
   * 评价分类学习函数 f(j) = sumi(ai * yi * Kij) + b
   * @param k
   * @return
   */
  def learnedFun(k: Int) = {
    var s = 0
    for(i <- 0 until trainNum) {
      if(alpha(i) > 0)
        s += alpha(i) * target(i) * kernelFunc(i, k)
    }
    s -= b
    s
  }

  /**
   * 评价分类学习函数2 f(j) = sumi(ai * yi * Kij) + b
   * @param k
   * @return
   */
  def learnedFun2(k: DoubleMatrix) = {
    var s = 0
    for(i <- 0 until trainNum) {
      if(alpha(i) > 0)
        s += alpha(i) * target(i) * kernelFunc(i, k)
    }
    s -= b
    s
  }

  /**
   * 优化两个乘子，成功返回 1，失败返回 0
   * @param i1
   * @param i2
   * @return
   */
  def takeStep(i1: Int, i2: Int): Int = {
    if(i1 == i2) // 不会优化两个同一样本
      return 0

    val alpha1 = alpha(i1) // 记录乘子alphai1的旧值
    val alpha2 = alpha(i2) // 记录乘子alphai2的旧值
    var a1 = 0.0 // 记录乘子alphai1的新值
    var a2 = 0.0 // 记录乘子alphai2的新值
    val y1 = target(i1)
    val y2 = target(i2)

    var E1 = 0.0
    var E2 = 0.0
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
      if(alpha(i) > 0 && alpha(i) < C)
        errorCache(i) += t1 * kernelFunc(i1, i) + t2 * kernelFunc(i2, i) - deltaB
    }
    errorCache(i1) = 0
    errorCache(i2) = 0
    alpha(i1) = a1
    alpha(i2) = a2
    1
  }

  /**
   * 1：在non-bound 乘子中寻找maximum fabs(E1-E2)的样本
   * @param i1
   * @param E1
   * @return
   */
  def examineFirstChoice(i1: Int, E1: Double): Int = {
    var i2 = -1
    var tmax = 0.0
    for(k <- 0 until trainNum) {
      if(alpha(k) > 0 && alpha(k) < C) {
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
    val k0 = Random.nextInt % trainNum
    for(k <- 0 until trainNum) {
      val i2 = (k0 + k) % trainNum
      if(alpha(i2) > 0 && alpha(i2) < C && takeStep(i1, i2) == 1)
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
    val k0 = Random.nextInt % trainNum
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
    val y1 = target(i1)
    val alpha1 = alpha(i1)
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
  def errorRate(): Unit ={
    var ac = 0
    println("----------------- 测试结果 -----------------");
    for(i <- trainNum until N) {
      val tar = learnedFun(i)
      if((tar > 0 && target(i) > 0) || tar < 0 && target(i) < 0 )
        ac += 1
    }
    val accuracy = ac.toDouble / (N - trainNum)
    println("精确度：" + accuracy * 100 + "%")
  }

  /**
   * 计算误差率2
   */
  def errorRate2(data: RDD[(Int, DoubleMatrix)]): Double = {
    data.cache()
    val n = data.count().toDouble
    var ac = 0
    println("----------------- 测试结果 -----------------")
    val accuracy = data.map { case (label, features) =>
      val tar = learnedFun2(features)
      if ((tar > 0 && label > 0) || (tar < 0 && label < 0))
        ac = 1
      ac
    }.reduce(_+_) / n

    println("精确度：" + accuracy * 100 + "%")
    accuracy
  }
*/
  //  def run(): SVMModel = {
  //    /**  
  //      * 以下两层循环，开始时检查所有样本，选择不符合 KKT 条件的两个乘子进行优化，选择成功，返回 1，否则，返回 0  
  //     * 所以成功了，numChanged 必然大于 0，从第二遍循环时，就不从整个样本中去寻找不符合 KKT 条件的两个乘子进行优化， 
  //     * 而是从边界的乘子中去寻找，因为非边界样本需要调整的可能性更大，边界样本往往不被调整而始终停留在边界上。  
  //      * 如果，没找到，再从整个样本中去找，直到整个样本中再也找不到需要改变的乘子为止，此时，算法结束。  
  //      */
  //    /*    var numChanged = 0  // number of alpha[i], alpha[j] pairs changed in a single step in the outer loop  
  //    var examineAll = 1  // flag indicating whether the outer loop has to be made on all the alpha[i]
  //    while(numChanged > 0 || examineAll == 1) {
  //      numChanged = 0
  //      if(examineAll == 1) {
  //        for(k <- 0 until trainNum)
  //          numChanged += examineExample(k) // 检查所有样本
  //      }
  //      else {
  //        for(k <- 0 until trainNum) {
  //          if(alpha(k) != 0 && alpha(k) != C)
  //            numChanged += examineExample(k) // 寻找所有非边界样本的lagrange乘子
  //        }
  //      }
  //      if(examineAll == 1)
  //        examineAll = 0
  //      else if(numChanged == 0) // else if(!numChanged)
  //        examineAll = 1
  //    }
  //
  //    // 存放训练后的参数
  //    var supportVectors = 0
  //    for(i <- 0 until trainNum) {
  //      if(alpha(i) > 0 && alpha(i) <= C)
  //        supportVectors += 1
  //    }
  //    println("================ supportVectors: ================ " + supportVectors)
  //
  ////    errorRate() // 测试
  ////    var supports = ArrayBuffer[(Int, DoubleMatrix)]()
  ////    for(i <- 0 until trainNum) {
  ////      if(alpha(i) > 0 && alpha(i) <= C)
  ////        ArrayBuffer += (i, new DoubleMatrix())
  ////    }
  //*/
  //    new SVMModel()
  //  }

  def run(): SVMModel = {
    new SVMModel()
  }
}



object SVMTest {
  /**
   * RBF训练
   * @param input 输入路径
   * @param C 惩罚因子
   * @param eps 松弛变量
   * @param tolerance 容忍度
   * @param gamma RBF中的gamma参数
   * @return
   */
  def train(input: RDD[LabeledPoint], C: Double, eps: Double, tolerance: Double, gamma: Double): SVMModel = {
//    new SVMTest(input, C, eps, tolerance, gamma).run()
    new SVMWithRBF(input, C, eps, tolerance, gamma).run()
  }

//  def test(input: RDD[LabeledPoint]): Double = {
//    val testData = input.map { line =>
//      val label = line.label.toInt
//      val features = new DoubleMatrix(1, line.features.size, line.features.toDense.toArray:_*)
//      (label, features)
//    }
//   new SVMTest(null, 0.0, 0.0, 0.0, 0.0).errorRate2(testData)
//  }

  def main(args: Array[String]) {
//    val input = "hdfs://tseg0:9010/user/tseg/saijinchen/sample_libsvm_data.txt"
    val input = "D:\\spark-1.4.1-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt"
    val C = 1.0 // 惩罚因子
    val eps = 1.0E-12 // 松弛变量
    val tolerance = 0.001 // 容忍度
    val gamma = 0.5;  //RBF Kernel Function的参数   g=Gamma = 1/2*Sigma.^2 (width of Rbf)
//    val twoSigmaSquared = 2.0 // 1/gamma RBF核函数中的参数

    val minPartitions = 1
    val numFeatures = 0
//    var densePoints = Array[Array[Double]]()
//    var target = Array[Int]()

    val conf = new SparkConf().setAppName("SVMTest Application")
    val sc = new SparkContext(conf)

    val training = SVMLoadFile.loadLibSVMFile(sc, input, minPartitions)
    println("数据的行数" + training.count())


//    val data= MLUtils.loadLibSVMFile(sc, input)
//    val splits = data.randomSplit(Array(0.6, 0.4), 11L)
//    val training = splits(0).cache()
//    val test = splits(1)

    val model = SVMTest.train(training, C, eps, tolerance, gamma)

//    val accuracy = SVMTest.test(test)
//    println("精确度：" + accuracy * 100 + "%")
  }
}



































