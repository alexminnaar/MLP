import java.text.DecimalFormat

import alex.MLP
import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.{Matchers, WordSpec}


class MLPSpec extends WordSpec with Matchers {

  "MLP" should {


    "compute the correct forward pass" in {

      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      val input = DenseVector(1.0, 0.0, 0.0)

      val (output, activations) = MLP.forwardPass(Vector(w12, w23), input)

      output should equal(DenseVector(0.3676098854895219))

    }

    "compute the correct backward pass" in {

      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      val input = DenseVector(1.0, 0.0, 0.0)

      val (output, activations) = MLP.forwardPass(Vector(w12, w23), input)

      val target = DenseVector(0.0)

      val der = MLP.backwardPass(target, activations, Vector(w12, w23))

      der should equal(Vector(DenseMatrix((-0.08545932055436989, -0.04995009770841746, -0.04027066039548713)),
        DenseMatrix((-0.0034189759440937167, -0.0, -0.0),
          (-0.016026368070347838, -0.0, -0.0))))

    }

    "compute correct outer product" in {

      val dv1 = DenseVector(1.0, 2.0)
      val dv2 = DenseVector(3.0, 4.0)

      val op = MLP.outerProd(dv1, dv2)

      op should equal(DenseMatrix((3.0, 6.0),
        (4.0, 8.0)))

    }

    "solve the xor problem" in {

      val xor = Vector((DenseVector(1.0, 0.0, 0.0), DenseVector(0.0)),
        (DenseVector(1.0, 1.0, 0.0), DenseVector(1.0)),
        (DenseVector(1.0, 0.0, 1.0), DenseVector(1.0)),
        (DenseVector(1.0, 1.0, 1.0), DenseVector(0.0)))

      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      var weights = Vector(w12, w23)


      val stepSize = 0.2
      val numIter = 5000

      //train network using xor examples
      for (i <- (0 to numIter)) {


        for (ex <- xor) {

          val (output, activations) = MLP.forwardPass(weights, ex._1)
          val der = MLP.backwardPass(ex._2, activations, weights)

          weights = MLP.sgd(weights, der.reverse, stepSize)
        }

      }

      //test xor function
      val (o1, a1) = MLP.forwardPass(weights, xor(0)._1)
      o1 should equal(DenseVector(0.06558841610114152))


      val (o2, a2) = MLP.forwardPass(weights, xor(1)._1)
      o2 should equal(DenseVector(0.9420490971682289))

      val (o3, a3) = MLP.forwardPass(weights, xor(2)._1)
      o3 should equal(DenseVector(0.9236073162109176))

      val (o4, a4) = MLP.forwardPass(weights, xor(3)._1)
      o4 should equal(DenseVector(0.056917630053021126))

    }


  }

}
