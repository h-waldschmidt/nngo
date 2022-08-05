package neural

import "gonum.org/v1/gonum/mat"

// the used data consists of float vectors
// this achieves maximum flexibility
// users can use one hot encoding, one dimensional vectors and n-dimensional vectors

// you can read the complete dataset and then later convert it into splitSet with splitDataSet()
type CompleteSet struct {
	data mat.Dense
}

// saves train and test data into struct
type SplitSet struct {
	trainData mat.Dense
	testData  mat.Dense
}

func NewCompleteSet(data [][]float64) *CompleteSet {
	var completeSet *CompleteSet

	return completeSet
}

func NewSplitSet(testData, trainData [][]float64) *SplitSet {
	var splitSet *SplitSet

	return splitSet
}

func newSplitSet(data [][]float64, splitRation float64) *SplitSet {
	var splitSet *SplitSet

	return splitSet
}

func (set *CompleteSet) splitDataSet() (SplitSet, error) {
	var splitSet SplitSet

	return splitSet, nil
}

type Network struct {
	layers             []Dense
	loss               lossFunc
	lossFuncDerivative lossFuncDerivative
}

func (dense *Network) predict(input mat.VecDense) {}

func (dense *Network) train() {}

func (dense *Network) evaluate() {}
