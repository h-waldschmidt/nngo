package neuralnet

import (
	"gonum.org/v1/gonum/floats"
)

// default methods that have to be implemented by layers
type Layer interface {
	forward(input float64) float64
	backward(outputGradient []float64, learning_rate float64) []float64
}

type BaseLayer struct {
	input  []float64
	output []float64
}

type DenseLayer struct {
	base    BaseLayer
	weights []float64
	bias    float64
}

func (d DenseLayer) forward(input []float64) float64 {
	d.base.input = input
	return floats.Dot(d.weights, input)
}

func (d DenseLayer) backward(outputGradient []float64, learning_rate float64) []float64 {

	return nil
}
