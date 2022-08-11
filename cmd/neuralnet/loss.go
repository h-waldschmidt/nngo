package neural

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Each constant represents an loss function and its derivative
// This makes the definition of neural networks easier
const (
	LossMse = 0
	LossMae = 1
)

// tuple of lossFunction and lossFunctionDerivative
type lossTuple struct {
	loss           lossFunc
	lossDerivative lossFuncDerivative
}

// get function tuple based on specification number
func getLossTuple(activationSpecs int) (lossTuple, error) {
	var funcs lossTuple
	if activationSpecs < 0 || activationSpecs > 2 {
		return funcs, fmt.Errorf("wrong specification")
	}

	if activationSpecs == LossMse {
		funcs = lossTuple{Mse, MseDerivative}
		return funcs, nil
	} else {
		funcs = lossTuple{Mae, MaeDerivative}
		return funcs, nil
	}
}

type lossFunc func(yTrue, yPred mat.VecDense) (float64, error)

type lossFuncDerivative func(yTrue, yPred mat.VecDense) (mat.VecDense, error)

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

func MaeDerivative(yTrue, yPred mat.VecDense) (mat.VecDense, error) {
	var ans mat.VecDense
	if yTrue.Len() != yPred.Len() {
		return ans, fmt.Errorf("vectors need to have the same dimensions")
	}

	// TODO: add derivative of Mae
	return yTrue, nil
}
