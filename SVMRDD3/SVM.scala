import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
 * Created by tseg on 2015/11/18.
 */
object SVM {

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
    new SVMWithRBF(input, C, eps, tolerance, gamma).run()
  }

  def main(args: Array[String]) {
    //    val input = "hdfs://tseg0:9010/user/tseg/saijinchen/sample_libsvm_data.txt"
//    val input = "D:\\spark-1.4.1-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt"
    val input = "D:\\spark-1.4.1-bin-hadoop2.6\\data\\mllib\\sample_svm_data.txt"
    val C = 1.0 // 惩罚因子
    val eps = 1.0E-12 // 松弛变量
    val tolerance = 0.001 // 容忍度
    val gamma = 0.5;  //RBF Kernel Function的参数   g=Gamma = 1/2*Sigma.^2 (width of Rbf)

    val minPartitions = 1

    val conf = new SparkConf().setAppName("SVM Application")
    val sc = new SparkContext(conf)

//    val training = SVMLoadFile.loadLibSVMFile(sc, input, minPartitions)
    val training = SVMLoadFile.loadSVMFile(sc, input, minPartitions)
    println("数据的行数" + training.count())



//    val data= MLUtils.loadLibSVMFile(sc, input)
//    val splits = data.randomSplit(Array(0.6, 0.4), 11L)
//    val training = splits(0).cache()
//    val test = splits(1)

    val model = SVM.train(training, C, eps, tolerance, gamma)

    //    val accuracy = SVMTest.test(test)
    //    println("精确度：" + accuracy * 100 + "%")
  }
}
