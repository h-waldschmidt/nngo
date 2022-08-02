package neural

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Mse(yTrue, yPred mat.VecDense) (float64, error) {
	if yTrue.Len() != yPred.Len() {
		return 0.0, fmt.Errorf("vectors need to have the same dimensions")
	}

	var sum float64
	for i := 0; i < yTrue.Len(); i++ {
		sum += math.Pow(yTrue.AtVec(i)-yPred.AtVec(i), 2)
	}

	return sum / float64(yTrue.Len()), nil
}

func MseDerivative(yTrue, yPred mat.VecDense) (mat.VecDense, error) {
	var ans mat.VecDense
	if yTrue.Len() != yPred.Len() {
		return ans, fmt.Errorf("vectors need to have the same dimensions")
	}

	yTrue.SubVec(&yPred, &yTrue)
	yTrue.ScaleVec(2, &yTrue)
	return yTrue, nil
}

func Mae(yTrue, yPred mat.VecDense) (float64, error) {
	if yTrue.Len() != yPred.Len() {
		return 0.0, fmt.Errorf("vectors need to have the same dimensions")
	}

	var sum float64
	for i := 0; i < yTrue.Len(); i++ {
		sum += math.Abs(yTrue.AtVec(i) - yPred.AtVec(i))
	}

	return sum / float64(yTrue.Len()), nil
}

// TODO: add derivative of Mae
