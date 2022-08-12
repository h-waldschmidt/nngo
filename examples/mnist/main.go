package main

import (
	"encoding/csv"
	"fmt"
	"log"
	neural "nngo/cmd/neuralnet"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// reading mnist from csv, because it is easier
	// source: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
	// examples/mnist/
	train := readCSVToSet("mnist_train.csv")
	test := readCSVToSet("mnist_test.csv")
	splitSet := neural.SplitSet{Train: train, Test: test}

	network, err := neural.NewNetwork(
		[][]int{{28 * 28, 40, 2}, {40, 10, 2}},
		0,
	)
	if err != nil {
		log.Fatal(err)
	}

	network.Train(&splitSet.Train, 100, 0.1)

	accuracy := network.EvaluateOneHot(&splitSet.Test)
	fmt.Printf("Accuracy on test data: %v", accuracy)
}

func readCSVToSet(filePath string) neural.Set {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}
	data := mat.NewDense(len(records[0])-1, len(records), nil)
	labels := mat.NewDense(10, len(records), nil)
	for i, record := range records {

		// ignore first line
		if i != 0 {
			for j, entry := range record {
				num, err := strconv.Atoi(entry)
				if err != nil {
					log.Fatal(err)
				}
				if j == 0 {
					labels.Set(num, i-1, 1)
				} else {
					data.Set(j-1, i-1, float64(num))
				}
			}
		}
	}
	return neural.Set{Data: *data, Labels: *labels}
}
