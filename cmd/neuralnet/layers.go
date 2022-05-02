package neuralnet

import (
	"gonum.org/v1/gonum/mat"
)

// default methods that have to be implemented by layers
type Layer interface {
	forward(input float64) float64
	backward(outputGradient []float64, learning_rate float64) []float64
}

type BaseLayer struct {
	input  mat.VecDense
	output mat.VecDense
}

type DenseLayer struct {
	base    BaseLayer
	weights mat.Dense
	bias    mat.VecDense
}

// constructor for DenseLayer
func NewDenseLayer(inputSize int, outputSize int) DenseLayer {

}

func (d DenseLayer) forward(input mat.VecDense) mat.VecDense {
	d.base.input = input
	var ans *mat.VecDense
	ans.MulVec(&d.weights, &input)
	return *ans
}

func (d DenseLayer) backward(outputGradient []float64, learning_rate float64) []float64 {

	return nil
}
