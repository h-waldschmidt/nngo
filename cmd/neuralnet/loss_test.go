package neuralnet

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/mat"
)

func TestMSENormal(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})

	t.Run("Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{1, 2, 3, 4})

		ans, err := Mse(*a, *b)
		if ans != 0 || err != nil {
			t.Error("Expected zero")
		}
	})

	t.Run("Not_Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{2, 3, 1, 4})

		ans, err := Mse(*a, *b)
		if ans != 1.5 || err != nil {
			t.Errorf("Expected: %v, Got: %v", 1.5, ans)
		}
	})
}

func TestMSEError(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	b := mat.NewVecDense(6, []float64{1, 2, 3, 4, 5, 6})

	ans, err := Mse(*a, *b)

	if ans != 0.0 || err == nil {
		t.Error("Expected error")
	}
}

func TestMSEDerivativeNormal(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})

	t.Run("Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{1, 2, 3, 4})
		expected := mat.NewVecDense(4, []float64{0, 0, 0, 0})

		ans, err := MseDerivative(*a, *b)
		if cmp.Equal(expected, ans) || err != nil {
			t.Error("Expected zero vector")
		}
	})

	t.Run("Not_Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{2, 3, 1, 4})
		expected := mat.NewVecDense(4, []float64{2, 2, 4, 0})

		ans, err := MseDerivative(*a, *b)
		if cmp.Equal(expected, ans) || err != nil {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})
}

func TestMSEDerivativeError(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	b := mat.NewVecDense(6, []float64{1, 2, 3, 4, 5, 6})

	expected := mat.NewVecDense(4, []float64{0, 0, 0, 0})
	ans, err := MseDerivative(*a, *b)

	if cmp.Equal(expected, ans) || err == nil {
		t.Error("Expected error")
	}
}
