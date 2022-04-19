package neuralnet

// default methods that have to be implemented by layers
type Layer interface {
	forward(input float64) float64
	backward(outputGradient []float64, learning_rate float64) []float64
}

type BaseLayer struct {
	input  float64
	output float64
}
