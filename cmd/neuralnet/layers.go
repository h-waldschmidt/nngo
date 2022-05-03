package neuralnet

import (
	"math/rand"

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
func NewDenseLayer(inputSize int, outputSize int) *DenseLayer {
	if inputSize <= 0 || outputSize <= 0 {
		panic("inputSize and outputSize must be greater than 0")
	}

	var denseLayer *DenseLayer

	input := mat.NewVecDense(inputSize, make([]float64, inputSize))

	for i := 0; i < inputSize; i++ {
		input.SetVec(i, rand.NormFloat64())
	}

	denseLayer.base.input = *input

	output := mat.NewVecDense(outputSize, make([]float64, outputSize))
	bias := mat.NewVecDense(outputSize, make([]float64, outputSize))
	weights := mat.NewDense(outputSize, inputSize, make([]float64, outputSize*inputSize))

	for i := 0; i < outputSize; i++ {
		output.SetVec(i, rand.NormFloat64())
		bias.SetVec(i, rand.NormFloat64())
		for j := 0; j < inputSize; j++ {
			weights.Set(i, j, rand.NormFloat64())
		}
	}

	denseLayer.base.output = *output
	denseLayer.weights = *weights
	denseLayer.bias = *bias

	return denseLayer
}

func (d *DenseLayer) forward(input mat.VecDense) mat.VecDense {
	d.base.input = input
	var ans *mat.VecDense
	ans.MulVec(&d.weights, &input)
	ans.AddVec(ans, &d.bias)
	return *ans
}

func (d *DenseLayer) backward(outputGradient mat.VecDense, learning_rate float64) []float64 {
	var weightsGradient *mat.Dense
	transpose := &d.base.input
	weightsGradient.Mul(&outputGradient, transpose.T())

	return nil
}
