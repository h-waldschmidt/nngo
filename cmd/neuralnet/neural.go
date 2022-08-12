package neural

import (
	"fmt"
	"math"
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
	Data   mat.Dense
	Labels mat.Dense
}

// saves train and test data into struct
type SplitSet struct {
	Train Set
	Test  Set
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
	for i := 0; i < set.Data.RawMatrix().Cols; i++ {
		j := rand.Intn(i + 1)

		// swap data
		cache := mat.Col(nil, i, &set.Data)
		set.Data.SetCol(i, mat.Col(nil, j, &set.Data))
		set.Data.SetCol(j, cache)

		// swap labels
		cache = mat.Col(nil, i, &set.Labels)
		set.Labels.SetCol(i, mat.Col(nil, j, &set.Labels))
		set.Labels.SetCol(j, cache)
	}

	// take len(data)*splitRatio percent of the last elements as test data
	splitIndex := set.Data.RawMatrix().Cols - int(math.Ceil(float64(set.Data.RawMatrix().Cols)*splitRatio))
	trainData := mat.NewDense(set.Data.RawMatrix().Rows,
		splitIndex,
		nil,
	)
	trainLabels := mat.NewDense(set.Labels.RawMatrix().Rows,
		splitIndex,
		nil,
	)
	testData := mat.NewDense(set.Data.RawMatrix().Rows,
		int(math.Ceil(float64(set.Data.RawMatrix().Cols)*splitRatio)),
		nil,
	)
	testLabels := mat.NewDense(set.Labels.RawMatrix().Rows,
		int(math.Ceil(float64(set.Data.RawMatrix().Cols)*splitRatio)),
		nil,
	)

	for i := 0; i < splitIndex; i++ {
		trainData.SetCol(i, mat.Col(nil, i, &set.Data))
		trainLabels.SetCol(i, mat.Col(nil, i, &set.Labels))
	}

	for i := splitIndex; i < set.Data.RawMatrix().Cols; i++ {
		testData.SetCol(i-splitIndex, mat.Col(nil, i, &set.Data))
		testLabels.SetCol(i-splitIndex, mat.Col(nil, i, &set.Labels))
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
func NewNetwork(layerSpecs [][]int, lossSpecs int) (*Network, error) {
	layers := make([]Dense, len(layerSpecs))
	for i, tuple := range layerSpecs {
		if len(tuple) != 3 {
			return nil, fmt.Errorf("unexpected layer tuple: %v", tuple)
		}
		var err error
		cache, err := NewDense(tuple[0], tuple[1], tuple[2])
		if err != nil {
			return nil, err
		}
		layers[i] = *cache
	}
	funcs, err := getLossTuple(lossSpecs)
	if err != nil {
		return nil, err
	}
	network := Network{layers, funcs.loss, funcs.lossDerivative}
	return &network, nil
}

func (dense *Network) predict(input mat.VecDense) mat.VecDense {
	for _, layer := range dense.layers {
		input = layer.forward(input)
	}
	return input
}

func (dense *Network) Train(train *Set, epochs int, learningRate float64) error {
	for i := 0; i < epochs; i++ {
		diff := 0.0
		for j := 0; j < train.Data.RawMatrix().Cols; j += 100 {
			out := dense.predict(GetColVector(&train.Data, j))
			cache, err := dense.loss(GetColVector(&train.Labels, j), out)
			if err != nil {
				return err
			}
			diff += cache

			grad, err := dense.lossDerivative(GetColVector(&train.Labels, j), out)
			if err != nil {
				return err
			}

			for k := range dense.layers {
				grad = dense.layers[len(dense.layers)-1-k].backward(grad, learningRate/float64(i+1))
			}
		}
		diff /= float64(train.Data.RawMatrix().Cols)
		diff *= 100
		fmt.Printf("Epoch = %v, Error = %v \n", i+1, diff)
	}
	return nil
}

// this evaluate function only works for one hot encoded input
func (dense *Network) EvaluateOneHot(test *Set) float64 {
	diff := 0.0
	for i := 0; i < test.Data.RawMatrix().Cols; i++ {
		input := GetColVector(&test.Data, i)
		predicted := dense.predict(input)

		predictedIndex := GetMaxIndex(&predicted)
		expectedOutput := GetColVector(&test.Labels, i)
		realIndex := GetMaxIndex(&expectedOutput)

		if predictedIndex == realIndex {
			diff++
		}
	}

	diff /= float64(test.Data.RawMatrix().Cols)
	return diff
}
