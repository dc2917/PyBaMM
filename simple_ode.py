import pybamm
import numpy as np

# Define variables
x_n = pybamm.Variable("Negative electrode stochiometry")
x_p = pybamm.Variable("Positive electrode stochiometry")

# Define parameters
x_n_0 = pybamm.Parameter("Initial negative electrode stochiometry")
x_p_0 = pybamm.Parameter("Initial positive electrode stochiometry")
Q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
Q_p = pybamm.Parameter("Positive electrode capacity [A.h]")
R = pybamm.Parameter("Electrode resistance [Ohm]")
I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
U_n = pybamm.FunctionParameter(
    "Negative electrode OCV", {"Negative electrode stochiometry": x_n}
)
U_p = pybamm.FunctionParameter(
    "Positive electrode OCV", {"Positive electrode stochiometry": x_p}
)

# Define the model
model = pybamm.BaseModel("Simple ODE battery model")
model.rhs = {
    x_n: -I / Q_n,
    x_p: -I / Q_p,
}
model.initial_conditions = {x_n: x_n_0, x_p: x_p_0}
model.variables["Voltage [V]"] = U_p - U_n - I * R
model.variables["Negative electrode stochiometry"] = x_n
model.variables["Positive electrode stochiometry"] = x_p

# Define events
model.events = [
    pybamm.Event("Stop at x_n = 0", x_n - 0),
    pybamm.Event("Stop at x_n = 1", 1 - x_n),
    pybamm.Event("Stop at x_p = 0", x_p - 0),
    pybamm.Event("Stop at x_p = 0", 1 - x_p),
]

# Define parameter values
def graphite_LGM50_ocp_Chen2020(sto):
    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )
    return u_eq

def nmc_LGM50_ocp_Chen2020(sto):
    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )
    return u_eq

parameter_values = pybamm.ParameterValues(
    {
        "Initial negative electrode stochiometry": 0.9,
        "Initial positive electrode stochiometry": 0.1,
        "Negative electrode capacity [A.h]": 1.0,
        "Positive electrode capacity [A.h]": 1.0,
        "Electrode resistance [Ohm]": 0.1,
        "Current function [A]": lambda t: 1 + 0.5 * pybamm.sin(100 * t),
        "Negative electrode OCV": graphite_LGM50_ocp_Chen2020,
        "Positive electrode OCV": nmc_LGM50_ocp_Chen2020,
    }
)

# Solve the model
sim = pybamm.Simulation(model, parameter_values=parameter_values)
soln = sim.solve([0, 1])
soln.plot(
    [
        "Voltage [V]",
        "Negative electrode stochiometry",
        "Positive electrode stochiometry"
    ]
)
