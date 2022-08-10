package neural

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/mat"
)

func TestNewSet(t *testing.T) {
	data := [][]float64{
		{1, 1, -4},
		{0, -2, 0},
		{0, 3, 0},
	}
	labels := [][]float64{
		{0, 1},
		{1, 0},
		{1, 0},
	}

	expectedData := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		1, -2, 3,
		-4, 0, 0,
	})
	expectedLabels := mat.NewDense(2, 3, []float64{
		0, 1, 1,
		1, 0, 0,
	})

	set, err := NewSet(data, labels)
	expectedSet := Set{*expectedData, *expectedLabels}

	if err != nil {
		t.Errorf("Didn't expect error. Got: %v", err)
	}

	if !cmp.Equal(*set, expectedSet, cmp.AllowUnexported(mat.Dense{}, Set{})) {
		t.Errorf("Expected: %v, Got: %v", set, expectedSet)
	}
}

func TestNewSetError(t *testing.T) {
	data := [][]float64{
		{1, 1, -4},
		{0, -2, 0},
		{0, 3, 0},
	}
	labels := [][]float64{
		{0, 1},
		{1, 0},
	}

	_, err := NewSet(data, labels)

	if err == nil {
		t.Error("Expected error.")
	}
}

func TestNewSplitSet(t *testing.T) {
	trainData := [][]float64{
		{1, 1, -4},
		{0, -2, 0},
		{0, 3, 0},
	}
	trainLabels := [][]float64{
		{0, 1},
		{1, 0},
		{1, 0},
	}
	testData := [][]float64{
		{1, 1, -4},
		{0, -2, 0},
		{0, 3, 0},
	}
	testLabels := [][]float64{
		{0, 1},
		{1, 0},
		{1, 0},
	}

	expectedTrainData := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		1, -2, 3,
		-4, 0, 0,
	})
	expectedTrainLabels := mat.NewDense(2, 3, []float64{
		0, 1, 1,
		1, 0, 0,
	})
	expectedTestData := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		1, -2, 3,
		-4, 0, 0,
	})
	expectedTestLabels := mat.NewDense(2, 3, []float64{
		0, 1, 1,
		1, 0, 0,
	})
	set, err := NewSplitSet(trainData, testData, trainLabels, testLabels)
	expectedSet := SplitSet{
		Set{*expectedTrainData, *expectedTrainLabels},
		Set{*expectedTestData, *expectedTestLabels},
	}

	if err != nil {
		t.Errorf("Didn't expect error. Got: %v", err)
	}

	if !cmp.Equal(*set, expectedSet, cmp.AllowUnexported(mat.Dense{}, SplitSet{}, Set{})) {
		t.Errorf("Expected: %v, Got: %v", set, expectedSet)
	}
}

func TestNewSplitSetError(t *testing.T) {
	t.Run("test mismatch", func(t *testing.T) {
		trainData := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		trainLabels := [][]float64{
			{0, 1},
			{1, 0},
		}
		testData := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		testLabels := [][]float64{
			{0, 1},
			{1, 0},
		}

		_, err := NewSplitSet(trainData, testData, trainLabels, testLabels)

		if err == nil {
			t.Error("Expected error.")
		}
	})

	t.Run("train mismatch", func(t *testing.T) {
		trainData := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		trainLabels := [][]float64{
			{0, 1},
			{1, 0},
			{1, 0},
		}
		testData := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		testLabels := [][]float64{
			{0, 1},
			{1, 0},
		}

		_, err := NewSplitSet(trainData, testData, trainLabels, testLabels)

		if err == nil {
			t.Error("Expected error.")
		}
	})
}

func TestNewSpitSetAlt(t *testing.T) {
	data := [][]float64{
		{1, 1, -4},
		{0, -2, 0},
		{0, 3, 0},
	}
	labels := [][]float64{
		{0, 1},
		{1, 0},
		{1, 0},
	}

	set, err := NewSplitSetAlt(data, labels, 0.2)

	if err != nil {
		t.Errorf("Didn't expect error. Got: %v", err)
	}

	if set.train.data.RawMatrix().Cols != 2 ||
		set.train.labels.RawMatrix().Cols != 2 ||
		set.test.data.RawMatrix().Cols != 1 ||
		set.test.labels.RawMatrix().Cols != 1 {
		t.Errorf("Didn't expect this split: %v", set)
	}
}

func TestNewSplitSetAltError(t *testing.T) {

	t.Run("size mismatch", func(t *testing.T) {
		data := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		labels := [][]float64{
			{0, 1},
			{1, 0},
		}

		_, err := NewSplitSetAlt(data, labels, 0.2)
		if err == nil {
			t.Error("Expected error.")
		}
	})

	t.Run("invalid splitRation", func(t *testing.T) {
		data := [][]float64{
			{1, 1, -4},
			{0, -2, 0},
			{0, 3, 0},
		}
		labels := [][]float64{
			{0, 1},
			{1, 0},
			{1, 0},
		}

		_, err := NewSplitSetAlt(data, labels, 2)
		if err == nil {
			t.Error("Expected error.")
		}
	})
}
