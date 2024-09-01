//
// Created by Maria on 8/27/2024.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <string>
#include "Tensor.h"
#include "Graph.h"

class ComputationalGraph;

/* ---------- abstract base class for operations ---------- */

class Operation {
private:
    const ComputationalGraph& graph_handle;
    std::vector<Operation*> parents;
    std::vector<Operation*> children;
    std::string name;
    FTensor value;
    bool is_mutable;

public:
    Operation(const ComputationalGraph& _graph_handle, bool is_mutable, std::string name)
    :
        graph_handle(_graph_handle),
        name(name),
        value(empty()),
        is_mutable(is_mutable)
    {}

    Operation(const ComputationalGraph& _graph_handle, FTensor value, bool is_mutable, std::string name)
    :
        graph_handle(_graph_handle),
        name(name),
        value(value),
        is_mutable(is_mutable)
    {}

    virtual ~Operation() = default;

    ComputationalGraph& get_graph_handle();
    std::vector<Operation*> get_parents();
    std::vector<Operation*> get_children();
    bool get_mutable();
    std::string get_name() {
        return name;
    }

    FTensor &get_value() {
        return value;
    }

    void set_value(FTensor data) {
        value = data;
    }

    virtual FTensor forward() = 0;
    virtual FTensor backward() = 0;
};

/* ---------- specific operations ---------- */

class Constant : public Operation {
public:
    Constant(FTensor value, const ComputationalGraph& graph_handle)
    :
        Operation(graph_handle, value, false, "Constant")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

class Placeholder : public Operation {
public:
    explicit Placeholder(const ComputationalGraph& graph_handle)
        : Operation(graph_handle, true, "Placeholder")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

class Variable : public Operation {
public:
    Variable(FTensor value, const ComputationalGraph& graph_handle)
        : Operation(graph_handle, value, true, "Constant")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

class Add : public Operation {
public:
    Add(Operation* left, Operation* right, const ComputationalGraph& graph_handle)
        : Operation(graph_handle, left->get_value() + right->get_value(), false, "Add")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

class Mul : public Operation {
public:
    Mul(Operation* left, Operation* right, const ComputationalGraph& graph_handle)
        : Operation(graph_handle, left->get_value() * right->get_value(), false, "Add")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

class Transpose : public Operation {
public:
    Transpose(Operation* operation, const ComputationalGraph& graph_handle)
        : Operation(graph_handle, operation->get_value().T(), false, "Transpose")
    {}

    FTensor forward() override {
        return get_value();
    }

    FTensor backward() override {
        return get_value();
    }
};

#endif //OPERATION_H
