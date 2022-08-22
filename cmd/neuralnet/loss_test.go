package neural

import (
	"math/rand"
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/mat"
)

func TestGetLossTuple(t *testing.T) {
	t.Run("MSE", func(t *testing.T) {
		tuple, err := getLossTuple(0)
		if err != nil {
			t.Errorf("Didn't expect error. Got: %v", err)
		}

		// create vector
		vecOne := mat.NewVecDense(3, nil)
		vecTwo := mat.NewVecDense(3, nil)
		for i := 0; i < 3; i++ {
			vecOne.SetVec(i, rand.Float64())
			vecTwo.SetVec(i, rand.Float64())
		}

		expected, err := Mse(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}
		ans, err := tuple.loss(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}

		if !cmp.Equal(expected, ans) {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}

		expectedVec, err := MseDerivative(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}
		ansVec, err := tuple.lossDerivative(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}

		if !cmp.Equal(expectedVec, ansVec, cmp.AllowUnexported(mat.VecDense{}, Set{})) {
			t.Errorf("Expected: %v, Got: %v", expectedVec, ansVec)
		}
	})

	t.Run("MAE", func(t *testing.T) {
		tuple, err := getLossTuple(1)
		if err != nil {
			t.Errorf("Didn't expect error. Got: %v", err)
		}

		// create vector
		vecOne := mat.NewVecDense(3, nil)
		vecTwo := mat.NewVecDense(3, nil)
		for i := 0; i < 3; i++ {
			vecOne.SetVec(i, rand.Float64())
			vecTwo.SetVec(i, rand.Float64())
		}

		expected, err := Mae(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}
		ans, err := tuple.loss(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}

		if !cmp.Equal(expected, ans) {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}

		expectedVec, err := MaeDerivative(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}
		ansVec, err := tuple.lossDerivative(*vecOne, *vecTwo)
		if err != nil {
			t.Error("Didn't expect error")
		}

		if !cmp.Equal(expectedVec, ansVec, cmp.AllowUnexported(mat.VecDense{}, Set{})) {
			t.Errorf("Expected: %v, Got: %v", expectedVec, ansVec)
		}
	})
}

func TestMSENormal(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})

	t.Run("Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{1, 2, 3, 4})

		ans, err := Mse(*a, *b)
		if ans != 0 || err != nil {
			t.Error("Expected zero")
		}
	})

	t.Run("NotZero", func(t *testing.T) {
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

	t.Run("NotZero", func(t *testing.T) {
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

func TestMAENormal(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})

	t.Run("Zero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{1, 2, 3, 4})

		ans, err := Mae(*a, *b)
		if ans != 0 || err != nil {
			t.Error("Expected zero")
		}
	})

	t.Run("NotZero", func(t *testing.T) {
		b := mat.NewVecDense(4, []float64{2, 3, 1, 4})

		ans, err := Mae(*a, *b)
		if ans != 1 || err != nil {
			t.Errorf("Expected: %v, Got: %v", 1, ans)
		}
	})
}

func TestMAEError(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	b := mat.NewVecDense(6, []float64{1, 2, 3, 4, 5, 6})

	ans, err := Mae(*a, *b)

	if ans != 0.0 || err == nil {
		t.Error("Expected error")
	}
}
