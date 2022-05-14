package neuralnet

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// default methods that have to be implemented by layers
type Layer interface {
	forward(input mat.VecDense) mat.VecDense
	backward(outputGradient mat.VecDense, learningRate float64) mat.Dense
}

type Base struct {
	input  mat.VecDense
	output mat.VecDense
}

type activationFunc func(float64) float64

func activationVector(vector mat.VecDense, activation activationFunc) mat.VecDense {
	for i := 0; i < vector.Len(); i++ {
		vector.SetVec(i, activation(vector.AtVec(i)))
	}

	return vector
}

type Activation struct {
	base           Base
	activationFunc activationFunc
	activationDer  activationFunc
}
type Dense struct {
	base                 Base
	weights              mat.Dense
	bias                 mat.VecDense
	activation           activationFunc
	activationDerivative activationFunc
}

// constructor for DenseLayer
func NewDense(inputSize, outputSize int, activation, activationDerivative activationFunc) *Dense {
	if inputSize <= 0 || outputSize <= 0 {
		panic("inputSize and outputSize must be greater than 0")
	}

	var dense *Dense
	dense.activation = activation
	dense.activationDerivative = activationDerivative

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

	// apply activation function / activaation layer
	cache := activationVector(*ans, d.activation)
	ans = &cache
	return *ans
}

func (d *Dense) backward(outputGradient mat.VecDense, learningRate float64) mat.Dense {
	// update activation layer
	cache := activationVector(d.base.input, d.activationDerivative)
	outputGradient.MulVec(&outputGradient, &cache)

	var weightsGradient *mat.Dense
	transpose := &d.base.input
	weightsGradient.Mul(&outputGradient, transpose.T())

	weightsGradient.Scale(learningRate, weightsGradient)
	d.weights.Sub(&d.weights, weightsGradient)

	cacheOutputGradient := outputGradient
	cacheOutputGradient.ScaleVec(learningRate, &cacheOutputGradient)
	d.bias.SubVec(&d.bias, &cacheOutputGradient)

	// type assertion needed since T() returns mat.Matrix not mat.Dense
	weightsTranspose := d.weights.T().(*mat.Dense)
	weightsTranspose.Mul(weightsTranspose, &outputGradient)
	return *weightsTranspose
}
