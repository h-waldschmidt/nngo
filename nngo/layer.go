package nngo

import (
	"fmt"
	"log"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// default methods that have to be implemented by each layer
// a layer needs an forward and backward propagation method
type Layer interface {
	forward(input mat.VecDense) mat.VecDense
	backward(outputGradient mat.VecDense, learningRate float64) mat.VecDense
}

// basic layer which consists of input and output vector
type Base struct {
	input  mat.VecDense
	output mat.VecDense
}

// Dense layer consists of a base layer with a weight matrix, bias vector and
// an activation function with its derivative
type Dense struct {
	base    Base
	weights mat.Dense
	bias    mat.VecDense
}

// constructor for DenseLayer
// creates a new dense layer with random values for the vectors and matrices with the given data
// inputSize and outputSize need to be positive
func NewDense(inputSize, outputSize int) (*Dense, error) {
	if inputSize <= 0 || outputSize <= 0 {
		return nil, fmt.Errorf("inputSize and outputSize must be greater than 0")
	}
	var dense Dense

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

	return &dense, nil
}

// forward propagation with the following formula:
// ans = activationFunc(weights * input + bias)
func (d *Dense) forward(input mat.VecDense) mat.VecDense {
	d.base.input = input
	var ans mat.VecDense
	ans.MulVec(&d.weights, &input)
	ans.AddVec(&ans, &d.bias)
	return ans
}

// backward propagation by using gradient descent
// nice explanation can be found here: https://www.youtube.com/watch?v=Ilg3gGewQ5U
func (d *Dense) backward(outputGradient mat.VecDense, learningRate float64) mat.VecDense {
	// calculate the outputGradient for the next layer
	weightsTranspose := mat.NewDense(
		d.weights.RawMatrix().Rows,
		d.weights.RawMatrix().Cols,
		nil,
	)
	weightsTranspose.Copy(&d.weights)
	var inputGradient mat.VecDense
	inputGradient.MulVec(weightsTranspose.T(), &outputGradient)

	// update weights
	var weightsGradient mat.Dense
	transpose := mat.NewDense(d.base.input.Len(), 1, nil)
	transpose.Copy(&d.base.input)
	weightsGradient.Mul(&outputGradient, transpose.T())
	weightsGradient.Scale(learningRate, &weightsGradient)
	d.weights.Sub(&d.weights, &weightsGradient)

	// update bias
	outputGradient.ScaleVec(learningRate, &outputGradient)
	d.bias.SubVec(&d.bias, &outputGradient)

	return inputGradient
}

type Activation struct {
	base                 Base
	activation           activationFunc
	activationDerivative activationFunc
}

func NewActivation(size, activationSpecs int) (*Activation, error) {
	if size <= 0 {
		return nil, fmt.Errorf("inputSize and outputSize must be greater than 0")
	}

	funcTuple, err := getActivationTuple(activationSpecs)
	if err != nil {
		return nil, err
	}

	var activation Activation
	activation.activation = funcTuple.activation
	activation.activationDerivative = funcTuple.activationDerivative

	input := mat.NewVecDense(size, make([]float64, size))
	output := mat.NewVecDense(size, make([]float64, size))

	for i := 0; i < size; i++ {
		input.SetVec(i, rand.NormFloat64())
		output.SetVec(i, rand.NormFloat64())
	}

	activation.base.input = *input
	activation.base.output = *output

	return &activation, nil
}

func (act *Activation) forward(input mat.VecDense) mat.VecDense {
	act.base.input = input
	return activationVector(input, act.activation)
}

func (act *Activation) backward(outputGradient mat.VecDense, learningRate float64) mat.VecDense {
	cache := activationVector(act.base.input, act.activationDerivative)
	outputGradient, err := componentWise(outputGradient, cache)
	if err != nil {
		log.Fatal(err)
	}
	return outputGradient
}

// applies the given activation function on each element of the vector
func activationVector(vector mat.VecDense, activation activationFunc) mat.VecDense {
	ans := mat.NewVecDense(vector.Len(), nil)
	ans.CopyVec(&vector)
	for i := 0; i < ans.Len(); i++ {
		ans.SetVec(i, activation(ans.AtVec(i)))
	}

	return *ans
}

func componentWise(a, b mat.VecDense) (mat.VecDense, error) {
	var ans mat.VecDense
	if a.Len() != b.Len() {
		return ans, fmt.Errorf("vectors need to have the same length")
	}
	ans = a
	for i := 0; i < a.Len(); i++ {
		ans.SetVec(i, a.AtVec(i)*b.AtVec(i))
	}
	return ans, nil
}
