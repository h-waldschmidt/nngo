package neural

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// provides function to that make accessability of the gonum.mat functions more accessible

// return the column as a vector instead of an slice
func GetColVector(matrix mat.Dense, index int) mat.VecDense {
	var col []float64
	col = mat.Col(col, index, &matrix)
	vector := mat.NewVecDense(len(col), col)
	return *vector
}

// returns the index of the element with the max value
func GetMaxIndex(vector mat.VecDense) int {
	maxValue := math.Inf(-1)
	maxIndex := -1
	for i := 0; i < vector.Len(); i++ {
		value := vector.AtVec(i)
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}

	return maxIndex
}
