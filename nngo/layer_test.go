package nngo

import (
	"math/rand"
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/mat"
)

func TestNewDenseNormal(t *testing.T) {
	dense, err := NewDense(4, 2)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	if dense.base.input.Len() != 4 || dense.base.output.Len() != 2 {
		t.Error("Input and output have not expected dimensions")
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
	dense, err := NewDense(-1, 2)
	if dense != nil || err == nil {
		t.Error("There should be an error")
	}
}

func TestDenseForwardNormal(t *testing.T) {
	dense, err := NewDense(4, 2)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	output := dense.forward(*input)
	if output.Len() != 2 {
		t.Error("Input and output have not expected dimensions")
	}
}

func TestDenseBackwardNormal(t *testing.T) {
	dense, err := NewDense(4, 2)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(2, []float64{1, 2})
	output := dense.backward(*input, 1)

	rows, cols := output.Dims()
	if rows != 4 || cols != 1 {
		t.Error("Weights matrix doesn't have expected dimensions")
	}
}

// TODO: add test and error handling for wrong inputs in forward and backward
// TODO: add testing for right range of values, e.g. gradient of backward
// TODO: add testing for helper functions

func TestNewActivationNormal(t *testing.T) {
	activation, err := NewActivation(4, 0)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	if activation.base.input.Len() != 4 || activation.base.output.Len() != 4 {
		t.Error("Input and output have not expected dimensions")
	}

	r := rand.Float64()
	if activation.activation(r) != Sigmoid(r) || activation.activationDerivative(r) != SigmoidDerivative(r) {
		t.Errorf("Expected: %v, Got: %v", Sigmoid(r), activation.activation(r))
	}

	r = rand.Float64()
	if activation.activationDerivative(r) != SigmoidDerivative(r) {
		t.Errorf("Expected: %v, Got: %v", SigmoidDerivative(r), activation.activationDerivative(r))
	}
}

func TestNewActivationError(t *testing.T) {
	t.Run("InvalidInputSize", func(t *testing.T) {
		activation, err := NewActivation(-1, 2)
		if activation != nil || err == nil {
			t.Error("Expected error.")
		}
	})

	t.Run("InvalidActivation", func(t *testing.T) {
		activation, err := NewActivation(3, 3)
		if activation != nil || err == nil {
			t.Error("Expected error.")
		}
	})
}

func TestActivationForwardNormal(t *testing.T) {
	activation, err := NewActivation(4, 2)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	output := activation.forward(*input)
	expected := activationVector(*input, Tanh)
	if !cmp.Equal(output, expected, cmp.AllowUnexported(mat.VecDense{})) {
		t.Errorf("Expected: %v, Got: %v", expected, output)
	}
}

func TestActivationBackwardNormal(t *testing.T) {
	activation, err := NewActivation(4, 2)

	if err != nil {
		t.Errorf("Didn't expect this error: %v", err)
	}

	input := mat.NewVecDense(2, []float64{1, 2})
	_ = activation.forward(*input)

	output := activation.backward(*input, 1)
	expected := activationVector(*mat.NewVecDense(2, []float64{1, 2}), TanhDerivative)
	expected.SetVec(1, expected.AtVec(1)*2)
	if !cmp.Equal(output, expected, cmp.AllowUnexported(mat.VecDense{})) {
		t.Errorf("Expected: %v, Got: %v", expected, output)
	}
}
