import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Created by tseg on 2015/11/18.
 */
object SVMLoadFile {
  def loadLibSVMFile(sc: SparkContext, path: String, minPartitions: Int): RDD[LabeledPoint] = {
    val data = sc.textFile(path, minPartitions) // 读取文件

    val N = data.count() // 样本数
    val parsed = data.map(_.trim()) // 消除每行两边的空格
        .filter(line => !(line.isEmpty || line.startsWith("#"))) // 排除空行和以#开头的行
        .map { line =>
        val items = line.split(' ') // 空格分隔处理过的每行，返回一个array
      val label = items.head.toInt  // 构造常量label存储行头并转换成Double
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          // 取数组尾部，对每一行进行下面的处理
          val indexAndValue = item.split(':') // 以:号分隔数据每个元素，存入indexAndValue
        val index = indexAndValue(0).toInt - 1 // 将向量下标修改为从0开始
        val value = indexAndValue(1).toDouble // 将向量转换成Double
          (index, value)
        }.unzip
        (label, indices.toArray, values.toArray) // 将获得的标签、向量下标、特征向量组成元组
      }

    // 获取数据维数（特征数）
    parsed.persist(StorageLevel.MEMORY_ONLY)
    val d = parsed.map { case (label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
    println("维数: " + d)
    // 最后，将元组转换成MLlib专用的LabeledPoint并返回
    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }
}
