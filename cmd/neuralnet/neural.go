package neural

import (
	"fmt"
	"math/rand"

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
// labels row size should match the output size of the output layer
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
// the size of each label vector should match the output size of the output layer
func NewSet(data, labels [][]float64) (*Set, error) {
	if len(data) != len(labels) {
		return nil, fmt.Errorf("size of data and labels should match")
	}

	// transform data and matrix into matrices
	dataMatrix := mat.NewDense(len(data[0]), len(data), nil)
	labelMatrix := mat.NewDense(len(labels[0]), len(labels), nil)

	for i := range data {
		dataMatrix.SetCol(i, data[i])
		labelMatrix.SetCol(i, labels[i])
	}

	set := Set{*dataMatrix, *labelMatrix}
	return &set, nil
}

// converts data vectors into a matrix
// each subslice of the data slice should represent a vector
// the size of each label vector should match the output size of the output layer
func NewSplitSet(testData, trainData, testLabels, trainLabels [][]float64) (*SplitSet, error) {
	if len(testData) != len(testLabels) {
		return nil, fmt.Errorf("size of test data and test labels should match")
	}
	if len(trainData) != len(trainLabels) {
		return nil, fmt.Errorf("size of train data and train labels should match")
	}

	// transform test data and labels into matrices
	testDataMatrix := mat.NewDense(len(testData[0]), len(testData), nil)
	testLabelMatrix := mat.NewDense(len(testLabels[0]), len(testLabels), nil)

	for i := range testData {
		testDataMatrix.SetCol(i, testData[i])
		testLabelMatrix.SetCol(i, testLabels[i])
	}

	// transform train data and labels into matrices
	trainDataMatrix := mat.NewDense(len(trainData[0]), len(trainData), nil)
	trainLabelMatrix := mat.NewDense(len(trainLabels[0]), len(trainLabels), nil)

	for i := range trainData {
		trainDataMatrix.SetCol(i, trainData[i])
		trainLabelMatrix.SetCol(i, trainLabels[i])
	}
	splitSet := SplitSet{Set{*trainDataMatrix, *trainLabelMatrix}, Set{*testDataMatrix, *testLabelMatrix}}

	return &splitSet, nil
}

// converts data vectors into a matrix
// each subslice of the data slice should represent a vector
// the size of each label vector should match the output size of the output layer
func NewSplitSetAlt(data, labels [][]float64, splitRatio float64) (*SplitSet, error) {
	if len(data) != len(labels) {
		return nil, fmt.Errorf("size of data and labels should match")
	}

	// transform data and matrix into matrices
	dataMatrix := mat.NewDense(len(data[0]), len(data), nil)
	labelMatrix := mat.NewDense(len(labels[0]), len(labels), nil)

	for i := range data {
		dataMatrix.SetCol(i, data[i])
		labelMatrix.SetCol(i, labels[i])
	}

	set := Set{*dataMatrix, *labelMatrix}
	splitSet, err := set.splitDataSet(splitRatio)
	if err != nil {
		return nil, err
	}

	return &splitSet, nil
}

// splits the data and label set into a training and test set
// splitRatio specifies the size of the test data
// splitRatio should be between 0 and 1
func (set *Set) splitDataSet(splitRatio float64) (SplitSet, error) {
	var splitSet SplitSet
	if splitRatio <= 0.0 || splitRatio >= 1.0 {
		return splitSet, fmt.Errorf("splitRatio should be a value between 0 and 1")
	}

	// shuffle data and labels
	for i := 0; i < set.data.RawMatrix().Cols; i++ {
		j := rand.Intn(i + 1)

		// swap data
		cache := mat.Col(nil, i, &set.data)
		set.data.SetCol(i, mat.Col(nil, j, &set.data))
		set.data.SetCol(j, cache)

		// swap labels
		cache = mat.Col(nil, i, &set.labels)
		set.labels.SetCol(i, mat.Col(nil, j, &set.labels))
		set.labels.SetCol(j, cache)
	}

	// take len(data)*splitRatio percent of the last elements as test data
	splitIndex := set.data.RawMatrix().Cols - int(float64(set.data.RawMatrix().Cols)*splitRatio)
	trainData := mat.NewDense(set.data.RawMatrix().Rows,
		splitIndex,
		nil,
	)
	trainLabels := mat.NewDense(set.labels.RawMatrix().Rows,
		splitIndex,
		nil,
	)
	testData := mat.NewDense(set.data.RawMatrix().Rows,
		int(float64(set.data.RawMatrix().Cols)*splitRatio),
		nil,
	)
	testLabels := mat.NewDense(set.labels.RawMatrix().Rows,
		int(float64(set.data.RawMatrix().Cols)*splitRatio),
		nil,
	)

	for i := 0; i < splitIndex; i++ {
		trainData.SetCol(i, mat.Col(nil, i, &set.data))
		trainLabels.SetCol(i, mat.Col(nil, i, &set.labels))
	}

	for i := splitIndex; i < set.data.RawMatrix().Cols; i++ {
		testData.SetCol(i, mat.Col(nil, i, &set.data))
		testLabels.SetCol(i, mat.Col(nil, i, &set.labels))
	}

	splitSet = SplitSet{Set{*trainData, *trainLabels}, Set{*testData, *testLabels}}
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

	for _, layer := range dense.layers {
		prediction = layer.forward(input)
		input = prediction
	}
	return prediction
}

func (dense *Network) train(trainData mat.Dense, epochs int) {}

func (dense *Network) evaluate(dataSet *SplitSet) float64 {
	return 0.0
}
