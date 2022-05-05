#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier FastSigmoidId = {"custom.ops", "FastSigmoid", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier FastSigmoidGradId = {"custom.ops", "FastSigmoidGrad",
                                                    1};
} // namespace CustomGradOperators

class FastSigmoidOp;
class FastSigmoidOpx;
class FastSigmoidGradOpx;

class FastSigmoidGradOp : public popart::Op {
public:
  FastSigmoidGradOp(const FastSigmoidOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<FastSigmoidGradOp>(*this);
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

class FastSigmoidOp : public popart::Op {
public:
  FastSigmoidOp(const popart::OperatorIdentifier &_opid, float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}
  FastSigmoidOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<FastSigmoidOp>(*this);
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
    upops.emplace_back(new FastSigmoidGradOp(*this));
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
      FastSigmoidOpDef({OpDefinition::Inputs({{"input", T}}),
                      OpDefinition::Outputs({{"output", T}}),
                      OpDefinition::Attributes()});

static popart::OpCreator<FastSigmoidOp> FastSigmoidOpCreator(
      popart::OpDefinitions({{CustomOperators::FastSigmoidId, FastSigmoidOpDef}}),
      [](const popart::OpCreatorInfo &info) {
        // default alpha is 10**(-2)
        float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
            "alpha", 1e-2f);
        return std::make_unique<FastSigmoidOp>(info.opid, alpha, info.settings);
        return std::make_unique<FastSigmoidOp>(info.opid, info.settings);
      },
      true);
} // namespace

namespace pe = popops::expr;

class FastSigmoidOpx : public popart::popx::Opx {
public:
  FastSigmoidOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<FastSigmoidOp>(
        op, {CustomOperators::FastSigmoidId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<FastSigmoidOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? 0:1
    auto expression = pe::Select(pe::Const(0.0f), pe::Const(1.0f),
                                 pe::Lt(pe::_1, pe::Const(0.0f)));


    popops::mapInPlace(graph(), expression, {input}, prog,
                       debugContext("FastSigmoid"), poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

class FastSigmoidGradOpx : public popart::popx::Opx {
public:
  FastSigmoidGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<FastSigmoidGradOp>(op, {CustomGradOperators::FastSigmoidGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<FastSigmoidGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    // pe::Mul expression = pe::Mul(pe::Select(pe::Const(0.0f), pe::Const(1.0f),
    //                                        pe::Lt(pe::_2, pe::Const(0.0f))),  // check if pe::_2 is less than 0.0. If so, return pe:Const(0.0f) else pe::Const(1.0f)
    //                            pe::_1);

    auto expression = pe::Divide(pe::_1, 
                                 pe::Pow(pe::Add(pe::Abs(pe::_2),pe::Const(1.0f)), pe::Const(2.0f)));                          

    auto output =
        popops::map(graph(), expression, {grad, input}, prog,
                    debugContext("FastSigmoidGrad"), poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

FastSigmoidGradOp::FastSigmoidGradOp(const FastSigmoidOp &fwdOp)
    : popart::Op(CustomGradOperators::FastSigmoidGradId, fwdOp.settings),
      alpha(fwdOp.getAlpha()) {}

const std::vector<popart::GradInOutMapper> &
FastSigmoidGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &FastSigmoidGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void FastSigmoidGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

void FastSigmoidGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

static popart::popx::OpxCreator<FastSigmoidOpx> FastSigmoidOpxCreator(
    {CustomOperators::FastSigmoidId});
static popart::popx::OpxCreator<FastSigmoidGradOpx>
    FastSigmoidGradOpxCreator({CustomGradOperators::FastSigmoidGradId});
