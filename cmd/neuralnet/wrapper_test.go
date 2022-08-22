package neural

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/mat"
)

func TestGetColVectorNormal(t *testing.T) {
	dense := mat.NewDense(4, 3, []float64{
		1, 0, 0,
		1, -2, 3,
		-4, 0, 0,
		0, 0, 0,
	})

	t.Run("firsVector", func(t *testing.T) {
		expected := *mat.NewVecDense(4, []float64{1, 1, -4, 0})

		ans := GetColVector(*dense, 0)
		if !cmp.Equal(expected, ans, cmp.AllowUnexported(mat.VecDense{})) {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})

	t.Run("secondVector", func(t *testing.T) {
		expected := *mat.NewVecDense(4, []float64{0, -2, 0, 0})

		ans := GetColVector(*dense, 1)
		if !cmp.Equal(expected, ans, cmp.AllowUnexported(mat.VecDense{})) {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})

	t.Run("thirdVector", func(t *testing.T) {
		expected := *mat.NewVecDense(4, []float64{0, 3, 0, 0})

		ans := GetColVector(*dense, 2)
		if !cmp.Equal(expected, ans, cmp.AllowUnexported(mat.VecDense{})) {
			t.Errorf("Expected: %v, Got: %v", expected, ans)
		}
	})
}
