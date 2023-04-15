import nest
import numpy as np

# Network parameters
num_excitatory = 432
num_inhibitory = 108
num_neurons = num_excitatory + num_inhibitory
neuron_model = "iaf_psc_alpha"

# Synaptic parameters
tau_syn_exc = 0.5  # Excitatory synaptic time constant in ms
tau_syn_inh = 0.5  # Inhibitory synaptic time constant in ms
w_exc = 1.0  # Excitatory synaptic weight
w_inh = -5.0  # Inhibitory synaptic weight

# STDP parameters
tau_plus = 16.8  # ms
tau_minus = 33.7  # ms
A_plus = 0.1
A_minus = 0.12
Wmax = 3.0

nest.ResetKernel()
nest.set_verbosity("M_WARNING")

# Set the simulation time resolution to 0.1 ms
nest.SetKernelStatus({"resolution": 1})

# Create neurons
neurons = nest.Create(neuron_model, num_neurons)

excitatory_neurons = neurons[:num_excitatory]
inhibitory_neurons = neurons[num_excitatory:]

# Set the parameters for excitatory and inhibitory neurons
nest.SetStatus(excitatory_neurons, {"tau_syn_ex": tau_syn_exc, "tau_syn_in": tau_syn_inh})
nest.SetStatus(inhibitory_neurons, {"tau_syn_ex": tau_syn_exc, "tau_syn_in": tau_syn_inh})

# Create STDP synapse model
stdp_triplet_synapse = nest.CopyModel(
    "stdp_triplet_synapse",
    "stdp_triplet_synapse_exc",
    {
        "A_plus": A_plus,
        "A_minus": A_minus,
        "tau_plus": tau_plus,
        "tau_minus": tau_minus,
        "Wmax": Wmax,
    },
)

# Connect neurons with STDP synapses
conn_dict = {"rule": "all_to_all"}

syn_dict_exc = {"model": "stdp_triplet_synapse_exc", "weight": w_exc, "delay": 1.0}
nest.Connect(excitatory_neurons, neurons, conn_dict, syn_dict_exc)

# Connect neurons with static inhibitory synapses
syn_dict_inh = {"weight": w_inh, "delay": 1.0}
nest.Connect(inhibitory_neurons, neurons, conn_dict, syn_dict_inh)

# Simulate the network
sim_time = 1000.0  # Simulation time in ms
nest.Simulate(sim_time)


# RNN class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Excitatory and inhibitory neuron models
        self.excitatory = nn.RNNCell(input_size, hidden_size)
        self.inhibitory = nn.RNNCell(input_size, hidden_size)
        
        # Synaptic facilitation and depression parameters
        self.exc_u = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.exc_x = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.inh_u = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.inh_x = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        
        # STDP parameters
        self.alpha = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(-0.001, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, h_exc, h_inh, u_exc, x_exc, u_inh, x_inh):
        # Excitatory neuron update
        h_exc = self.excitatory(input, h_exc)
        u_exc = (1 - self.theta) * u_exc + self.theta * h_exc
        x_exc = (1 + self.alpha * (u_exc - 1)) * x_exc
        x_exc = torch.clamp(x_exc, min=0.0, max=1.0)
        h_exc *= x_exc
        
        # Inhibitory neuron update
        h_inh = self.inhibitory(input, h_inh)
        u_inh = (1 - self.theta) * u_inh + self.theta * h_inh
        x_inh = (1 + self.alpha * (u_inh - 1)) * x_inh
        x_inh = torch.clamp(x_inh, min=0.0, max=1.0)
        h_inh *= x_inh
        
        # Short-term synaptic plasticity update
        z_exc = (1 - u_exc) * torch.exp(-input / self.tau)
        z_inh = (1 - u_inh) * torch.exp(-input / self.tau)
        u_exc = u_exc * torch.exp(-z_exc) + self.gamma * z_exc
        u_inh = u_inh * torch.exp(-z_inh) + self.gamma * z_inh
        
        # STDP update
        delta_w = self.beta * torch.exp(-torch.abs(self.tau)) * torch.mm(h_exc.t(), h_inh)
        delta_w *= (self.delta + torch.exp(self.lambda_ * delta_w) * (self.theta - self.delta))
        self.output.weight += delta_w
"""
