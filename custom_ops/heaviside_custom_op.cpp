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
const popart::OperatorIdentifier HeavisideId = {"custom.ops", "Heaviside", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier HeavisideGradId = {"custom.ops", "HeavisideGrad",
                                                    1};
} // namespace CustomGradOperators

class HeavisideOp;
class HeavisideOpx;
class HeavisideGradOpx;

class HeavisideGradOp : public popart::Op {
public:
  HeavisideGradOp(const HeavisideOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<HeavisideGradOp>(*this);
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

class HeavisideOp : public popart::Op {
public:
  HeavisideOp(const popart::OperatorIdentifier &_opid, float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}
  HeavisideOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<HeavisideOp>(*this);
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
    upops.emplace_back(new HeavisideGradOp(*this));
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
      HeavisideOpDef({OpDefinition::Inputs({{"input", T}}),
                      OpDefinition::Outputs({{"output", T}}),
                      OpDefinition::Attributes()});

static popart::OpCreator<HeavisideOp> HeavisideOpCreator(
      popart::OpDefinitions({{CustomOperators::HeavisideId, HeavisideOpDef}}),
      [](const popart::OpCreatorInfo &info) {
        // default alpha is 10**(-2)
        float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
            "alpha", 1e-2f);
        return std::make_unique<HeavisideOp>(info.opid, alpha, info.settings);
        return std::make_unique<HeavisideOp>(info.opid, info.settings);
      },
      true);
} // namespace

namespace pe = popops::expr;

class HeavisideOpx : public popart::popx::Opx {
public:
  HeavisideOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<HeavisideOp>(
        op, {CustomOperators::HeavisideId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<HeavisideOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? 0:1
    auto expression = pe::Select(pe::Const(0.0f), pe::Const(1.0f),
                                 pe::Lt(pe::_1, pe::Const(0.0f)));


    popops::mapInPlace(graph(), expression, {input}, prog,
                       debugContext("Heaviside"), poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

class HeavisideGradOpx : public popart::popx::Opx {
public:
  HeavisideGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<HeavisideGradOp>(op, {CustomGradOperators::HeavisideGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<HeavisideGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    pe::Mul expression = pe::Mul(pe::Select(pe::Const(0.0f), pe::Const(1.0f),
                                            pe::Lt(pe::_2, pe::Const(0.0f))),
                                 pe::_1);

    auto output =
        popops::map(graph(), expression, {grad, input}, prog,
                    debugContext("HeavisideGrad"), poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

HeavisideGradOp::HeavisideGradOp(const HeavisideOp &fwdOp)
    : popart::Op(CustomGradOperators::HeavisideGradId, fwdOp.settings),
      alpha(fwdOp.getAlpha()) {}

const std::vector<popart::GradInOutMapper> &
HeavisideGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &HeavisideGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void HeavisideGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

void HeavisideGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

static popart::popx::OpxCreator<HeavisideOpx> HeavisideOpxCreator(
    {CustomOperators::HeavisideId});
static popart::popx::OpxCreator<HeavisideGradOpx>
    HeavisideGradOpxCreator({CustomGradOperators::HeavisideGradId});
