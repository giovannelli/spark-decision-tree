name := "Gender Project"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1"

libraryDependencies ++= Seq(
  "org.apache.spark"  % "spark-core_2.10"              % "2.0.1" % "provided",
  "org.apache.spark"  % "spark-mllib_2.10"             % "2.0.1"
  )