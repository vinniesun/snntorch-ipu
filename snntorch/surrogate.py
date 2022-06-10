import torch
import ctypes
import os
import popart
import poptorch

# Spike-gradient functions


#so_path_sigmoid = "./so_file/sigmoid_custom_ops.so"
#if not os.path.isfile(so_path_sigmoid):
#    print("Missing Sigmoid Custom Operation File")
#    exit(1)
#ctypes.cdll.LoadLibrary(so_path_sigmoid)


#def build_and_run_sigmoid(input_data, run_on_ipu=True):
#    y = poptorch.custom_op(
#            [input_data],
#            "Sigmoid",
#            "custom.ops",
#            1,
#            example_outputs=[input_data],
#    )
#    return y[0]


class StraightThroughEstimator:
    """
    Straight Through Estimator.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Straight Through Estimator.

        .. math::

                \\frac{∂S}{∂U}=1


    """

    cwd = os.path.dirname(__file__)
    so_path_ste = os.path.join(cwd, "so_file/straight_through_estimator_custom_ops.so")
    #so_path_ste = "./so_file/straight_through_estimator_custom_ops.so"
    if not os.path.isfile(so_path_ste):
        print("Missing Straight Through Estimator Custom Operation file!")
        print(so_path_ste)
        exit(1)
    ctypes.cdll.LoadLibrary(so_path_ste)

    def build_and_run_ste(input_data, run_on_ipu=True):
        y = poptorch.custom_op(
                [input_data],
                "StraightThroughEstimator",
                "custom.ops",
                1,
                example_outputs=[input_data],
        )
        return y[0]

def straight_through_estimator():
    """Straight Through Estimator surrogate gradient enclosed with a parameterized slope."""
    return StraightThroughEstimator.build_and_run_ste


class FastSigmoid:
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{U}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{(1+k|U|)^2}

    :math:`k` defaults to 25, and can be modified by calling ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    cwd = os.path.dirname(__file__)
    so_path_fast_sigmoid = os.path.join(cwd, "so_file/fast_sigmoid_custom_ops.so")
    #so_path_fast_sigmoid = "./so_file/fast_sigmoid_custom_ops.so"
    if not os.path.isfile(so_path_fast_sigmoid):
        print("Missing Fast Sigmoid  Custom Operation File")
        exit(1)
    ctypes.cdll.LoadLibrary(so_path_fast_sigmoid)


    def build_and_run_fast_sigmoid(input_data, run_on_ipu=True):
        y = poptorch.custom_op(
                [input_data],
                "FastSigmoid",
                "custom.ops",
                1,
                example_outputs=[input_data],
        )
        return y[0]

def fast_sigmoid():
    """FastSigmoid surrogate gradient."""
    return FastSigmoid.build_and_run_fast_sigmoid
