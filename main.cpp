#include <iostream>
#include "Tensor.h"
#include "Operation.h"

int main() {

	ComputationalGraph* graph = new ComputationalGraph();
	Variable* beta = new Variable(zeros({3, 1}), *graph);
	Variable* intercept = new Variable(zeros({3, 1}), *graph);
	Constant* X = new Constant(zeros({3, 3}), *graph);
	Mul* model_no_intercept = new Mul(X, beta, *graph);
	Add* model = new Add(model_no_intercept, intercept, *graph);

	delete graph;
	delete beta;
	delete intercept;
	delete X;
	delete model_no_intercept;
	delete model;

	return 0;
}
