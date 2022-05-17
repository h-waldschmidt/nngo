package neuralnet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func mse(yTrue, yPred mat.VecDense) float64 {
	if yTrue.Len() != yPred.Len() {
		panic("vectors need to have the same dimesions")
	}

	var sum float64
	for i := 0; i < yTrue.Len(); i++ {
		sum += math.Pow(yTrue.AtVec(i)-yPred.AtVec(i), 2)
	}

	return sum / float64(yTrue.Len())
}
