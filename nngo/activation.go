package nngo

import (
	"fmt"
	"math"
)

// Each constant represents an activation function and its derivative
// This makes the definition of neural networks easier
const (
	ActivationSigmoid = 0
	ActivationRelu    = 1
	ActivationTanh    = 2
)

// tuple of activationFunction and activationFunctionDerivative
type activationTuple struct {
	activation           activationFunc
	activationDerivative activationFunc
}

// get function tuple based on specification number
func getActivationTuple(activationSpecs int) (activationTuple, error) {
	var funcs activationTuple
	if activationSpecs < 0 || activationSpecs > 2 {
		return funcs, fmt.Errorf("wrong specification")
	}

	if activationSpecs == ActivationSigmoid {
		funcs = activationTuple{Sigmoid, SigmoidDerivative}
		return funcs, nil
	} else if activationSpecs == ActivationRelu {
		funcs = activationTuple{Relu, ReluDerivative}
		return funcs, nil
	} else {
		funcs = activationTuple{Tanh, TanhDerivative}
		return funcs, nil
	}
}

// activationFunc is a function that takes a float as input and has float as output
type activationFunc func(float64) float64

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

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhDerivative(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
