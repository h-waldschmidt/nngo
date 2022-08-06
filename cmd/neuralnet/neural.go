package neural

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// saves all the data and labels
//
// the used data consists of float vectors
// this achieves maximum flexibility
// users can use one hot encoding, one dimensional vectors and n-dimensional vectors
//
// each index of the data corresponds to the same label index in the labels data
// labels are saved as vectors for easier access during the training process
//
// you can read the complete dataset and then later convert it into splitSet with splitDataSet()
type Set struct {
	data   mat.Dense
	labels mat.Dense
}

// saves train and test data into struct
type SplitSet struct {
	train Set
	test  Set
}

// converts data vectors into a matrix
// each subslice of the data slice should represent a vector
func NewSet(data [][]float64, labels []float64, outputSize int) *Set {
	var set *Set

	return set
}

// converts data vectors into a matrix
// each subslice of the data slice should represent a vector
func NewSplitSet(testData, trainData [][]float64, testLabels, trainLabels []float64, outputSize int) *SplitSet {
	var splitSet *SplitSet

	return splitSet
}

// converts data vectors into a matrix
// each subslice of the data slice should represent a vector
func newSplitSet(data [][]float64, labels []float64, outputSize int, splitRatio float64) *SplitSet {
	var splitSet *SplitSet

	return splitSet
}

// splits the data and label set into a training and test set
// splitRatio specifies the size of the test
// splitRatio should be between 0 and 1
func (set *Set) splitDataSet(splitRatio float64) (SplitSet, error) {
	var splitSet SplitSet
	if splitRatio <= 0.0 || splitRatio >= 1.0 {
		return splitSet, fmt.Errorf("splitRatio should be a value between 0 and 1")
	}

	return splitSet, nil
}

// specifies a neural network
// layers are saved inside a slice
// this structure allows for almost every possible neural network configuration
type Network struct {
	layers         []Dense
	loss           lossFunc
	lossDerivative lossFuncDerivative
}

// create a neural network
// each layer is specified by a subslice
// e.g. {4, 5, 0} specifies a layer with 4 input, 5 output neurons and Sigmoid as a activation function
func NewNetwork(layerSpecs [][]int, lossSpecs int) *Network {
	return nil
}

func (dense *Network) predict(input mat.VecDense) mat.VecDense {
	var prediction mat.VecDense

	return prediction
}

func (dense *Network) train(trainData mat.Dense, epochs int) {}

func (dense *Network) evaluate(dataSet *SplitSet) float64 {
	return 0.0
}
