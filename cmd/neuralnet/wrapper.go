package neural

import "gonum.org/v1/gonum/mat"

// provides function to that make accessability of the gonum.mat functions more accessible

// return the column as a vector instead of an slice
func getColVector(matrix *mat.Dense, index int) mat.VecDense {
	var col []float64
	col = mat.Col(col, index, matrix)
	vector := mat.NewVecDense(len(col), col)
	return *vector
}
