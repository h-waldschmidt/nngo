package neuralnet

// default methods that have to be implemented by layers
type Layer interface {
}

type BaseLayer struct {
	input  float64
	output float64
}
