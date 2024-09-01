//
// Created by Maria on 8/31/2024.
//

#include "Graph.h"

void ComputationalGraph::append_operation(Operation* operation) {
    if (operation->get_name() == "PlaceholderNode") {
        placeholders.push_back(operation);
    }
    else {
        operations.push_back(operation);
    }
}

FTensor ComputationalGraph::execute(Operation* node, std::map<Operation*, FTensor> feed_dict) {

    if (operations.empty()) {
        return empty();
    }

    // Set all the placeholders
    /*
    for (int i = 0; i < placeholders.size(); i++) {
        placeholders[i]->set_value(feed_dict[placeholders[i]]);
    }
    */

    return node->forward();
}
