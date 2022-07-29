package neuralnet

// the used data consists of float vectors
// this achieves maximum flexibility
// users can use one hot encoding, one dimensional vectors and n-dimensional vectors

// you can read the complete dataset and then later convert it into splitSet with splitDataSet()
type CompleteSet struct {
	data [][]float64
}

// saves train and test data into struct
type SplitSet struct {
	trainData [][]float64
	testData  [][]float64
}

func (set *CompleteSet) splitDataSet() (SplitSet, error) {}

type DenseNeuralNet struct {
	layers []Dense
	cost   costFunc
}

func (dense *DenseNeuralNet) train() {}

func (dense *DenseNeuralNet) evaluate() {}
