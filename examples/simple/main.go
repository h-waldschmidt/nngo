package main

import (
	"fmt"
	"log"

	"github.com/h-waldschmidt/nngo/nngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// initialize test and train data with labels
	trainData := mat.NewDense(2, 4, []float64{0, 0, 1, 1, 0, 1, 0, 1})
	trainLabels := mat.NewDense(2, 4, []float64{0, 0, 1, 1, 0, 1, 0, 1})
	testData := mat.NewDense(2, 4, []float64{0, 0, 1, 1, 0, 1, 0, 1})
	testLabels := mat.NewDense(2, 4, []float64{0, 0, 1, 1, 0, 1, 0, 1})

	splitSet := nngo.SplitSet{
		Test:  nngo.Set{Data: *trainData, Labels: *trainLabels},
		Train: nngo.Set{Data: *testData, Labels: *testLabels},
	}

	// create the network
	network, err := nngo.NewNetwork(
		[][]int{
			{2, 10, 2}, // input layer with input size of 2, output size of 10 and tanh as activation function
			{10, 2, 2}, // output layer with input size of 10, output size of 2 and tanh as activation function
		},
		0, // loss function is MSE
	)
	if err != nil {
		log.Fatal(err)
	}

	// train the network with 1000 epochs
	network.Train(&splitSet.Train, 100, 0.1)

	// evaluate the performance of the network
	accuracy := network.EvaluateOneHot(&splitSet.Test)
	fmt.Printf("Accuracy on test data: %v", accuracy)
}
