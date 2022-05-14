package neuralnet

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func ReluDerivative(x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return 1
	}
}
