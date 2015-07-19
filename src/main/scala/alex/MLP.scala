package alex

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector}
import breeze.numerics.sigmoid

object MLP {

  /**
   * Forward pass to produce output for given input
   * @param weights weight matrices corresponding to each layer
   * @param input input vector
   * @return output vector
   */
  def forwardPass(weights: Vector[DenseMatrix[Double]],
                  input: DenseVector[Double]): (DenseVector[Double], Vector[DenseVector[Double]]) = {

    var b = input
    var activations: Vector[DenseVector[Double]] = Vector(b)

    for (weight <- weights.zipWithIndex) {

      val a = sigmoid(weight._1 * b)


      //val bNoBias=sigmoid(a)

      //Add bias to b unless it is the last layer
      if (weight._2 != weights.length - 1) {
        b = DenseVector.vertcat(DenseVector(1.0), a)
      }
      else {
        b = a
      }
      activations :+= b

    }

    (b, activations)

  }

  /**
   * Outer-product for two vectors
   */
  def outerProd(v1: DenseVector[Double], v2: DenseVector[Double]): DenseMatrix[Double] = {

    var newV1: DenseMatrix[Double] = DenseMatrix(v1.toArray)

    while (newV1.rows != v2.length) {
      newV1 = DenseMatrix.vertcat(newV1, v1.toDenseMatrix)
    }

    val bc = newV1(::, *) *= v2
    bc.underlying
  }

  /**
   * Backpropagation algorithm to obtain loss derivatives wrt each weight
   * @param target actual output
   * @param activations activations from forward pass
   * @param weights weight matrices corresponding to layers
   * @return weight derivative matrices.
   */
  def backwardPass(target: DenseVector[Double],
                   activations: Vector[DenseVector[Double]],
                   weights: Vector[DenseMatrix[Double]]): Vector[DenseMatrix[Double]] = {

    val prediction = activations.last
    var delta = prediction :* (prediction.map(x => (1 - x))) :* (target - prediction)

    var derivatives: Vector[DenseMatrix[Double]] = Vector.empty

    val reversedActivations = activations.dropRight(1).reverse
    val reversedWeights = weights.reverse

    for (act <- reversedActivations.zipWithIndex) {

      val derivative = outerProd(delta, act._1)

      derivatives :+= derivative.t

      val w = reversedWeights(act._2).delete(0, Axis._1)
      val dw = w.t * delta

      delta = sigmoidPrime(act._1(1 to -1)) :* (dw)

    }

    derivatives
  }


  def sigmoidPrime(a: DenseVector[Double]) = a.map(x => x * (1 - x))


  def sgd(weights: Vector[DenseMatrix[Double]],
          derivatives: Vector[DenseMatrix[Double]],
          stepSize: Double): Vector[DenseMatrix[Double]] = {

    var newWeights: Vector[DenseMatrix[Double]] = Vector.empty

    for (weight <- weights.zipWithIndex) {

      newWeights :+= weight._1 + (derivatives(weight._2) * stepSize)

    }
    newWeights
  }
}
