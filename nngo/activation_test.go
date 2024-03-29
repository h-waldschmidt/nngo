package nngo

import (
	"math"
	"math/rand"
	"testing"
)

func TestGetActivationTuple(t *testing.T) {
	t.Run("Sigmoid", func(t *testing.T) {
		tuple, err := getActivationTuple(0)
		if err != nil {
			t.Errorf("Didn't expect error. Got: %v", err)
		}

		r := rand.Float64()
		if Sigmoid(r) != tuple.activation(r) {
			t.Errorf("Expected: %v, Got: %v", Sigmoid(r), tuple.activation(r))
		}

		if SigmoidDerivative(r) != tuple.activationDerivative(r) {
			t.Errorf("Expected: %v, Got: %v", SigmoidDerivative(r), tuple.activationDerivative(r))
		}
	})

	t.Run("Relu", func(t *testing.T) {
		tuple, err := getActivationTuple(1)
		if err != nil {
			t.Errorf("Didn't expect error. Got: %v", err)
		}

		r := rand.Float64()
		if Relu(r) != tuple.activation(r) {
			t.Errorf("Expected: %v, Got: %v", Relu(r), tuple.activation(r))
		}

		if ReluDerivative(r) != tuple.activationDerivative(r) {
			t.Errorf("Expected: %v, Got: %v", ReluDerivative(r), tuple.activationDerivative(r))
		}
	})

	t.Run("Tanh", func(t *testing.T) {
		tuple, err := getActivationTuple(2)
		if err != nil {
			t.Errorf("Didn't expect error. Got: %v", err)
		}

		r := rand.Float64()
		if Tanh(r) != tuple.activation(r) {
			t.Errorf("Expected: %v, Got: %v", Tanh(r), tuple.activation(r))
		}

		if TanhDerivative(r) != tuple.activationDerivative(r) {
			t.Errorf("Expected: %v, Got: %v", TanhDerivative(r), tuple.activationDerivative(r))
		}
	})
}

func TestGetActivationTupleError(t *testing.T) {
	_, err := getActivationTuple(100)
	if err == nil {
		t.Error("Expected error.")
	}
}

func TestSigmoid(t *testing.T) {
	t.Run("Normal", func(t *testing.T) {
		expected := 1 / (1 + math.Exp(-10))
		ans := Sigmoid(10)
		if ans != expected {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})

	t.Run("OneHalf", func(t *testing.T) {
		ans := Sigmoid(0)
		if ans != 0.5 {
			t.Errorf("Expected: %v, Got: %v", 0.5, ans)
		}
	})

	t.Run("One", func(t *testing.T) {
		ans := Sigmoid(math.Inf(1))
		if ans != 1 {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})
}

func TestSigmoidDerivative(t *testing.T) {
	t.Run("Normal", func(t *testing.T) {
		expected := Sigmoid(10) * (1 - Sigmoid(10))
		ans := SigmoidDerivative(10)
		if ans != expected {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})

	t.Run("Zero", func(t *testing.T) {
		ans := SigmoidDerivative(math.Inf(1))
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})
}

func TestRelu(t *testing.T) {
	t.Run("NormalPositive", func(t *testing.T) {
		ans := Relu(10)
		if ans != 10 {
			t.Errorf("Expected: %v, Got: %v", 10, ans)
		}
	})

	t.Run("NormalNegative", func(t *testing.T) {
		ans := Relu(-10)
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})

	t.Run("NegativeInfinity", func(t *testing.T) {
		ans := Relu(math.Inf(-1))
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})

	t.Run("PositiveInfinity", func(t *testing.T) {
		ans := Relu(math.Inf(1))
		if ans != math.Inf(1) {
			t.Errorf("Expected: %v, Got: %v", math.Inf(1), ans)
		}
	})
}

func TestReluDerivative(t *testing.T) {
	t.Run("NormalPositive", func(t *testing.T) {
		ans := ReluDerivative(10)
		if ans != 1 {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})

	t.Run("NormalNegative", func(t *testing.T) {
		ans := ReluDerivative(-10)
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})

	t.Run("Zero", func(t *testing.T) {
		ans := ReluDerivative(0)
		if ans != 1 {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})
}

func TestTanh(t *testing.T) {
	t.Run("Zero", func(t *testing.T) {
		ans := Tanh(0)
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})

	t.Run("NegativeInfinity", func(t *testing.T) {
		ans := Tanh(math.Inf(-1))
		if ans != -1 {
			t.Errorf("Expected: %v, Got: %v", -1, ans)
		}
	})

	t.Run("PositiveInfinity", func(t *testing.T) {
		ans := Tanh(math.Inf(1))
		if ans != 1 {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})
}

func TestTanhDerivative(t *testing.T) {
	t.Run("Zero", func(t *testing.T) {
		ans := TanhDerivative(0)
		if ans != 1 {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})

	t.Run("NegativeInfinity", func(t *testing.T) {
		ans := TanhDerivative(math.Inf(-1))
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})

	t.Run("PositiveInfinity", func(t *testing.T) {
		ans := TanhDerivative(math.Inf(1))
		if ans != 0 {
			t.Errorf("Expected: %v, Got: %v", 0, ans)
		}
	})
}
