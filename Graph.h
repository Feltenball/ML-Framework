//
// Created by Maria on 8/27/2024.
//

#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include "Operation.h"
#include "Tensor.h"

class Operation;
class Placeholder;

class ComputationalGraph {
private:
    /*
    The graph splits up operations and placeholders into two different vectors.
    The reason for doing this is that the graph needs to be able to fill out placeholder using the feed_dict.
    Separating usual operations from placeholders allows us to enter feed_dict into the placeholders in a convenient way.
    */
    std::vector<Operation*> operations;
    std::vector<Operation*> placeholders;

public:
    ComputationalGraph() = default;
    ~ComputationalGraph() = default;

    FTensor execute(Operation* node, std::map<Operation*, FTensor> feed_dict);
    void append_operation(Operation* operation);
};

#endif //GRAPH_H
