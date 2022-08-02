package neural

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewDenseNormal(t *testing.T) {
	dense, err := NewDense(4, 2, Sigmoid, SigmoidDerivative)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	if dense.base.input.Len() != 4 || dense.base.output.Len() != 2 {
		t.Error("Input and output have not expected dimensions")
	}

	if dense.activation == nil || dense.activationDerivative == nil {
		t.Error("Expected activation function and derivative to be not nil")
	}

	if dense.bias.Len() != 2 {
		t.Error("Bias vector has not been initialized correctly")
	}

	rows, cols := dense.weights.Dims()
	if rows != 2 || cols != 4 {
		t.Error("Weights matrix doesn't have expected dimensions")
	}

}

func TestNewDenseError(t *testing.T) {
	dense, err := NewDense(-1, 2, Sigmoid, SigmoidDerivative)
	if dense != nil || err == nil {
		t.Error("There should be an error")
	}
}

func TestForwardNormal(t *testing.T) {
	dense, err := NewDense(4, 2, Sigmoid, SigmoidDerivative)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	output := dense.forward(*input)
	if output.Len() != 2 {
		t.Error("Input and output have not expected dimensions")
	}

	if output.AtVec(0) <= 0 || output.AtVec(0) > 1 || output.AtVec(1) <= 0 || output.AtVec(1) > 1 {
		t.Error("Didn't expect this kind of output")
	}
}

func TestBackwardNormal(t *testing.T) {
	dense, err := NewDense(4, 2, Sigmoid, SigmoidDerivative)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(2, []float64{1, 2})
	output := dense.backward(*input, 1)

	rows, cols := output.Dims()
	if rows != 2 || cols != 4 {
		t.Error("Weights matrix doesn't have expected dimensions")
	}
}

// TODO: add test and error handling for wrong inputs in forward and backward
// TODO: add testing for right range of values, e.g. gradient of backward
// TODO: add testing for helper functions
