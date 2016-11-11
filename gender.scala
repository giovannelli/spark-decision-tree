import org.apache.spark.rdd.RDD
// Import classes for MLLib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object GenderApp {
// define the User Schema
case class User(gender: String, firstname: String, country: String, language: String)

def getCurrentDirectory = new java.io.File( "." ).getCanonicalPath


def keyForValue(map: Map[String, Int], value: Int) = {
  val revMap = map map {_.swap}
  val key = revMap(value)
  key
}

// function to parse input into User class
def parseUser(str: String): User = {
  val line = str.split(",")
  User(line(0), line(1), line(2), line(3))
}

// load the data into a  RDD
val textRDD = sc.textFile(getCurrentDirectory +"/test.txt")
// MapPartitionsRDD[1] at textFile 

// parse the RDD of csv lines into an RDD of flight classes 
val usersRDD = textRDD.map(parseUser).cache()
println(usersRDD.first())

var countryMap: Map[String, Int] = Map()
var index: Int = 0
usersRDD.map(user => user.country).distinct.collect.foreach(x => { countryMap += (x -> index); index += 1 })
println(countryMap.toString)

var languageMap: Map[String, Int] = Map()
var index: Int = 0
usersRDD.map(user => user.language).distinct.collect.foreach(x => { languageMap += (x -> index); index += 1 })
println(languageMap.toString)

var nameMap: Map[String, Int] = Map()
var index: Int = 0
usersRDD.map(user => user.firstname).distinct.collect.foreach(x => { nameMap += (x -> index); index += 1 })
println(nameMap.toString)

var genderMap: Map[String, Int] = Map()
var index: Int = 0
usersRDD.map(user => user.gender).distinct.collect.foreach(x => { genderMap += (x -> index); index += 1 })
println(genderMap.toString)


//- Defining the features array
val mlprep = usersRDD.map(user => {
  val gender = genderMap(user.gender)
  val firstname = nameMap(user.firstname) 
  val country = countryMap(user.country) 
  val language = languageMap(user.language) 
  Array(gender.toDouble, firstname.toDouble, country.toDouble, language.toDouble)
})
mlprep.take(1)

//Making LabeledPoint of features - this is the training data for the model
val mldata = mlprep.map(x => LabeledPoint(x(0), Vectors.dense(x(1), x(2), x(3))))
//mldata.take(1)
//res7: Array[org.apache.spark.mllib.regression.LabeledPoint] = Array((0.0,[0.0,2.0,900.0,1225.0,6.0,385.0,214.0,294.0]))

// mldata0 is %85 not male users
val mldata0 = mldata.filter(x => x.label == 0.0).randomSplit(Array(0.85, 0.15))(1)
// mldata1 is %100 female users
val mldata1 = mldata.filter(x => x.label != 0.0)
// mldata2 is male and  female
val mldata2 = mldata0 ++ mldata1

//  split mldata2 into training and test data
val splits = mldata2.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

println(testData.take(1).label)

for ( x <- testData ) { println( x.label }
for ( x <- testData ) { println( x.features ) }


var categoricalFeaturesInfo = Map[Int, Int]()
//categoricalFeaturesInfo += (0 -> nameMap.size)
//categoricalFeaturesInfo += (1 -> countryMap.size)
//categoricalFeaturesInfo += (2 -> languageMap.size)


val numClasses = 3
// Defning values for the other parameters
val impurity = "gini"
val maxDepth = 9
val maxBins = 100000

// call DecisionTree trainClassifier with the trainingData , which returns the model
val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// print out the decision tree
model.toDebugString

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
labelAndPreds.take(3)


val wrongPrediction =(labelAndPreds.filter{
  case (label, prediction) => ( label !=prediction) 
  })

wrongPrediction.count()

val ratioWrong=wrongPrediction.count().toDouble/testData.count()

//res21: Array[org.apache.spark.mllib.regression.LabeledPoint] = Array((0.0,[18.0,6.0,900.0,1225.0,6.0,385.0,214.0,294.0]))
}