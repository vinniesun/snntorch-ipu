// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to create a custom operator for PopART, in this
// case a Leaky ReLU op that returns `x` for any element `x >= 0` and `x *
// alpha` for any element `x < 0`, where `alpha` is provided as a scalar
// attribute to the operator.
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier StraightThroughEstimatorId = {"custom.ops", "StraightThroughEstimator", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier StraightThroughEstimatorGradId = {"custom.ops", "StraightThroughEstimatorGrad",
                                                    1};
} // namespace CustomGradOperators

class StraightThroughEstimatorOp;
class StraightThroughEstimatorOpx;
class StraightThroughEstimatorGradOpx;

class StraightThroughEstimatorGradOp : public popart::Op {
public:
  StraightThroughEstimatorGradOp(const StraightThroughEstimatorOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<StraightThroughEstimatorGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  float getAlpha() const { return alpha; }

  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override;

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

private:
  float alpha;
};

class StraightThroughEstimatorOp : public popart::Op {
public:
  StraightThroughEstimatorOp(const popart::OperatorIdentifier &_opid, float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}
  StraightThroughEstimatorOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<StraightThroughEstimatorOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new StraightThroughEstimatorGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  // Attributes
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

namespace {
using popart::OpDefinition;
using popart::DataType;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
      StraightThroughEstimatorOpDef({OpDefinition::Inputs({{"input", T}}),
                      OpDefinition::Outputs({{"output", T}}),
                      OpDefinition::Attributes()});

static popart::OpCreator<StraightThroughEstimatorOp> StraightThroughEstimatorOpCreator(
      popart::OpDefinitions({{CustomOperators::StraightThroughEstimatorId, StraightThroughEstimatorOpDef}}),
      [](const popart::OpCreatorInfo &info) {
        // default alpha is 10**(-2)
        float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
            "alpha", 1e-2f);
        return std::make_unique<StraightThroughEstimatorOp>(info.opid, alpha, info.settings);
        return std::make_unique<StraightThroughEstimatorOp>(info.opid, info.settings);
      },
      true);
} // namespace

namespace pe = popops::expr;

class StraightThroughEstimatorOpx : public popart::popx::Opx {
public:
  StraightThroughEstimatorOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<StraightThroughEstimatorOp>(
        op, {CustomOperators::StraightThroughEstimatorId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<StraightThroughEstimatorOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x > 0.0f ? 1:0
    auto expression = pe::Select(pe::Const(1.0f), pe::Const(0.0f),
                                 pe::Gt(pe::_1, pe::Const(0.0f)));


    popops::mapInPlace(graph(), expression, {input}, prog,
                       debugContext("StraightThroughEstimator"), poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

class StraightThroughEstimatorGradOpx : public popart::popx::Opx {
public:
  StraightThroughEstimatorGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<StraightThroughEstimatorGradOp>(op, {CustomGradOperators::StraightThroughEstimatorGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<StraightThroughEstimatorGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    auto expression = pe::_1;

    auto output =
        popops::map(graph(), expression, {grad, input}, prog,
                    debugContext("StraightThroughEstimatorGrad"), poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

StraightThroughEstimatorGradOp::StraightThroughEstimatorGradOp(const StraightThroughEstimatorOp &fwdOp)
    : popart::Op(CustomGradOperators::StraightThroughEstimatorGradId, fwdOp.settings),
      alpha(fwdOp.getAlpha()) {}

const std::vector<popart::GradInOutMapper> &
StraightThroughEstimatorGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &StraightThroughEstimatorGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void StraightThroughEstimatorGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

void StraightThroughEstimatorGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

static popart::popx::OpxCreator<StraightThroughEstimatorOpx> StraightThroughEstimatorOpxCreator(
    {CustomOperators::StraightThroughEstimatorId});
static popart::popx::OpxCreator<StraightThroughEstimatorGradOpx>
    StraightThroughEstimatorGradOpxCreator({CustomGradOperators::StraightThroughEstimatorGradId});
