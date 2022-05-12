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

type Base struct {
	input  mat.VecDense
	output mat.VecDense
}

type Dense struct {
	base    Base
	weights mat.Dense
	bias    mat.VecDense
}

// constructor for DenseLayer
func NewDenseLayer(inputSize int, outputSize int) *Dense {
	if inputSize <= 0 || outputSize <= 0 {
		panic("inputSize and outputSize must be greater than 0")
	}

	var dense *Dense

	input := mat.NewVecDense(inputSize, make([]float64, inputSize))

	for i := 0; i < inputSize; i++ {
		input.SetVec(i, rand.NormFloat64())
	}

	dense.base.input = *input

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

	dense.base.output = *output
	dense.weights = *weights
	dense.bias = *bias

	return dense
}

func (d *Dense) forward(input mat.VecDense) mat.VecDense {
	d.base.input = input
	var ans *mat.VecDense
	ans.MulVec(&d.weights, &input)
	ans.AddVec(ans, &d.bias)
	return *ans
}

func (d *Dense) backward(outputGradient mat.VecDense, learning_rate float64) mat.Dense {
	var weightsGradient *mat.Dense
	transpose := &d.base.input
	weightsGradient.Mul(&outputGradient, transpose.T())

	weightsGradient.Scale(learning_rate, weightsGradient)
	d.weights.Sub(&d.weights, weightsGradient)

	cacheOutputGradient := outputGradient
	cacheOutputGradient.ScaleVec(learning_rate, &cacheOutputGradient)
	d.bias.SubVec(&d.bias, &cacheOutputGradient)

	// type assertion needed since T() returns mat.Matrix not mat.Dense
	weightsTranspose := d.weights.T().(*mat.Dense)
	weightsTranspose.Mul(weightsTranspose, &outputGradient)
	return *weightsTranspose
}

type Activation struct {
	base Base
}
