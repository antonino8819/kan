"""
Decodes a KAN model with 3 hidden layers max, from a given GA individual.
Experiments cover max 2 hidden layers so far. Thus, KAN architecture should
be [l1,l2,0].
"""


import shutil
import time
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import os

import torch

from scipy.optimize import curve_fit

from kan.MultKAN import MultKAN
from kan.spline import coef2curve


def run_decoder(n_ins, n_outs, arch, arg_gr, arg_k, arg_pr_thr, seed, test_name):
    """
    Runs the KAN decoder.

    :param n_ins: The number of input nodes.
    :param n_outs: The number of output nodes.
    :param arch: The KAN architecture.
    :param arg_gr: The number of each KAN spline grid knots.
    :param arg_k: The KAN splines degree.
    :param arg_pr_thr: The pruning threshold.
    :param seed: The simulation seed.
    :param test_name: The test name.
    """

    def create_ind(w, gr, k_val, pr_thr):
        """
        Creates a test GA individual
        :param w: KAN width (layers structure).
        :type w: list
        :param gr: Number of spline grid nodes.
        :type gr: int
        :param k_val: Splines grade.
        :type k_val: int
        :param pr_thr: Pruning threshold.
        :type pr_thr: float
        :return: The test individual.
        :rtype: list
        """
        ind = [w[0], w[1], w[2], gr, k_val, pr_thr] # individual without spline coeffs
        cod_spline = np.round(np.random.uniform(0, 1, 300), 3).tolist() # spline coeffs
        ind.extend(cod_spline)# complete individual, i.e. without spline coeffs
        return ind

    def kan_decode(ind):
        """
        Creates a KAN model from the GA test individual

        :param ind: The GA individual.
        :type ind: list

        :return: The KAN model.
        :rtype: MultKAN
        """
        # Decode KAN architecture
        width = [ind[0], ind[1], ind[2]] # KAN architecture without inputs and outputs
        eff_width = [] # complete KAN architecture, i.e. with inputs and outputs
        for each in width:
            if each == 0:
                break
            else:
                eff_width.append(each)

        # Calculate the number of needed spline coefficients
        layers = [inputs] + eff_width + [outputs]
        total_coefficients = 0 # number of spline coefficients fot the whole KAN
        for l in range(len(layers) - 1):
            total_coefficients += layers[l] * layers[l + 1] * (ind[3] + ind[4])

        def generate_coeffs(num_coeffs):
            """
            Original approximate technique for generating KAN coeffs without curse of dimensionality.
            """
            base_weights = np.random.uniform(0, 1, num_coeffs)
            m_classes = 300
            interval_size = round(1 / m_classes, 6)
            classes = []
            class_indices = []
            for i in range(m_classes):
                lower_bound = round(i * interval_size, 3)
                upper_bound = round((i + 1) * interval_size, 3)
                class_elements = base_weights[(base_weights >= lower_bound) & (base_weights < upper_bound)]
                indices = np.where((base_weights >= lower_bound) & (base_weights < upper_bound))[0]
                classes.append(class_elements)
                class_indices.append(indices)
            transformed_classes = []
            transformed_indices = []
            individual_alphas = individual[-m_classes:]
            for alpha, class_elements, indices in zip(individual_alphas, classes, class_indices):
                if alpha < 0.5:
                    transformed =  [round(alpha * element, 3) for element in class_elements]
                else:
                    transformed = [round(2 * (1 - alpha) * element + 2 * alpha - 1, 3) for element in class_elements]
                transformed_classes.append(transformed)
                transformed_indices.append(indices)
            all_transformed_values = np.concatenate(transformed_classes)
            all_transformed_indices = np.concatenate(transformed_indices)
            coeffs = deepcopy(base_weights)
            for i,t in zip(all_transformed_indices, all_transformed_values):
                coeffs[all_transformed_indices[i]] = all_transformed_values[i]
            coeffs.tolist()
            distances = [abs(w-base_w) for w, base_w in zip(coeffs, base_weights)]
            mean_distance = round(sum(distances) / len(distances), 3)
            file_name = "coeffs_m_dist.json"
            file_path = os.path.join(results_path, file_name)
            with open(file_path, "w") as file:
                json.dump(mean_distance, file)
            control_weights = np.random.uniform(0, 1, num_coeffs)
            distances_1 = [abs(w - base_w) for w, base_w in zip(control_weights, base_weights)]
            mean_distance_1 = round(sum(distances_1) / len(distances_1), 3)
            file_name = "contr_uni_m_dist.json"
            file_path = os.path.join(results_path, file_name)
            with open(file_path, "w") as file:
                json.dump(mean_distance_1, file)
            plt.figure()
            plt.title("Generated coefficients vs random base coefficients")
            x_coeffs = range(len(coeffs))
            x_base_weights = range(len(base_weights))
            plt.scatter(x_coeffs, coeffs, label='Gen. weights')
            plt.scatter(x_base_weights, base_weights, label='Base weights')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            fig_name = "gen_coeffs.png"
            fig_path = os.path.join(results_path, fig_name)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            plt.figure()
            plt.title("New random coefficients vs random base coefficients")
            x_coeffs = range(len(control_weights))
            x_base_weights = range(len(base_weights))
            plt.scatter(x_coeffs, control_weights, label='Contr. weights')
            plt.scatter(x_base_weights, base_weights, label='Base weights')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            fig_name = "contr_coeffs.png"
            fig_path = os.path.join(results_path, fig_name)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            return coeffs

        # Organize KAN coeffs to be added to the model
        coeffs = generate_coeffs(total_coefficients) # KAN coeffs
        organized_weights = [] # KAN coeffs well organized for being added to the model
        start_idx = 0
        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            num_weights = input_dim * output_dim * (ind[3] + ind[4])
            layer_weights = coeffs[start_idx:start_idx + num_weights]
            organized_weights.append(
                torch.tensor(layer_weights).reshape(input_dim, output_dim, ind[3] + ind[4])
            )
            start_idx += num_weights

        # Apply pruning
        pr_thr = ind[5]
        organized_weights = [
            torch.where(
                torch.mean(weight, dim=-1, keepdim=True) > pr_thr,
                weight,
                torch.tensor(0.0)
            )
            for weight in organized_weights
        ]
        # print(f'weights = {organized_weights}')

        def create_kan(input_dim=inputs,
                       output_dim=outputs,
                       width=None,
                       grid=ind[3],
                       k=ind[4],
                       base_fun = None,
                       custom_weights=None,
                       noise_scale=0.1, scale_base_mu=0.0, scale_base_sigma=1.0, grid_eps=0.02,
                       sp_trainable=True, sb_trainable=True, affine_trainable=False,
                       symbolic_enabled=True, sparse_init=False):
            """
            Creates a KAN with configurable parameters and allows manual weight initialization.

            :param input_dim: Dimension of the input.
            :type input_dim: int
            :param output_dim: Dimension of the output.
            :type output_dim: int
            :param width: Number of nodes per layer.
            :type width: list[int]
            :param grid: Number of intervals in the grid. Recommended range: 5-20.
            :type grid: int
            :param k: Order of the spline. Recommended range: 1-5.
            :type k: int
            :param base_fun: Base function for activations (e.g., 'silu').
            :type base_fun: str
            :param custom_weights: List of tensors to initialize weights.
            :type custom_weights: list[torch.Tensor]
            :param noise_scale: Initial noise scale. Recommended range: 0.0-0.5.
            :type noise_scale: float
            :param scale_base_mu: Mean for base function initialization. Recommended range: -1.0 to 1.0.
            :type scale_base_mu: float
            :param scale_base_sigma: Standard deviation for base function initialization. Recommended range: 0.0-1.0.
            :type scale_base_sigma: float
            :param grid_eps: Grid parameter (0: adaptive, 1: uniform). Recommended range: 0.0-1.0.
            :type grid_eps: float
            :param sp_trainable: Whether spline parameters are trainable.
            :type sp_trainable: bool
            :param sb_trainable: Whether base function parameters are trainable.
            :type sb_trainable: bool
            :param affine_trainable: Whether affine parameters are trainable.
            :type affine_trainable: bool
            :param symbolic_enabled: Enables symbolic computations.
            :type symbolic_enabled: bool
            :param sparse_init: Enables sparse initialization of weights.
            :type sparse_init: bool

            :returns: Created model.
            :rtype: MultKAN
            """
            # Create the KAN model
            model = MultKAN(
                width=[input_dim] + width + [output_dim],
                grid=grid,
                k=k,
                base_fun=base_fun,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                grid_eps=grid_eps,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable,
                affine_trainable=affine_trainable,
                symbolic_enabled=symbolic_enabled,
                sparse_init=sparse_init
            )
            # Compute the model on the CPU, instead of the GPU
            model.to('cpu')

            # Set KAN coeffs
            if custom_weights:
                for i, layer in enumerate(model.act_fun):
                    if hasattr(layer, 'coef'):
                        expected_shape = layer.coef.shape
                        if custom_weights[i].shape != expected_shape:
                            raise ValueError(
                                f"Custom weights for layer {i} must have shape {expected_shape}, but got {custom_weights[i].shape}")
                        with torch.no_grad():
                            layer.coef.copy_(custom_weights[i])
                            # print(f'KAN coeffs: layer {layer} -> {custom_weights[i]}')
            else:
                raise Exception("KAN coeffs not set!")

            # Override torch.std() to avoid warnings for small tensors while calculating std. dev.
            def safe_std(tensor, dim=0):
                return torch.std(tensor, dim=dim, unbiased=False) if tensor.numel() > 1 else torch.zeros_like(tensor)
            # Replace problematic std calculations in model
            for i, layer in enumerate(model.act_fun):
                if hasattr(layer, 'subnode_actscale'):
                    layer.subnode_actscale = safe_std(layer.subnode_actscale)
                if hasattr(layer, 'input_range'):
                    layer.input_range = safe_std(layer.input_range)
                if hasattr(layer, 'output_range_spline'):
                    layer.output_range_spline = safe_std(layer.output_range_spline)
                if hasattr(layer, 'output_range'):
                    layer.output_range = safe_std(layer.output_range)

            return model

        return create_kan(inputs,
                          outputs,
                          eff_width,
                          ind[3],
                          ind[4],
                          'silu',
                          organized_weights)

    def test_kan(model, input_vector):
        """
        Performs a forward pass on a KAN model with an input vector and returns node outputs as a dictionary.
        In other words, it evaluates the model by using the original KAN library.

        :param model: Created KAN model.
        :type model: MultKAN
        :param input_vector: Input vector.
        :type input_vector: torch.Tensor

        :returns: A tuple containing the model output and a dictionary where keys are spline labels and values are node outputs.
        :rtype: tuple
        """

        model.eval()  # set the model to evaluation mode
        node_outputs_dict = {} # KAN nodes with their values
        # Ensure the input matches the expected size
        if input_vector.shape[1] != model.width_in[0]:
            raise ValueError(f"Dimension mismatch: input_vector has {input_vector.shape[1]} features, expected {model.width_in[0]}.")

        # Add input nodes to the dictionary
        for node_idx, input_value in enumerate(input_vector.T):
            spline_label = f"IN{node_idx}"
            node_outputs_dict[spline_label] = input_value.tolist()

        # Get KAN output and node values
        with torch.no_grad():
            current_input = input_vector
            output = model(input_vector) # KAN output
            for layer_idx, layer in enumerate(model.act_fun):
                if hasattr(layer, 'coef'):
                    layer_output, _, _, _ = layer.forward(current_input)
                    current_input = layer_output
                    for node_idx, node_output in enumerate(layer_output.T):
                        node_label = f"L{layer_idx}N{node_idx}"
                        node_outputs_dict[node_label] = node_output.tolist()

        return output, node_outputs_dict

    def check_continuity(model):
        """
        Check if all outputs are connected to the rest of the network and
        at least one input->all outputs path exists in the KAN model.

        :param model: KAN model
        :type model: MultKAN
        """
        import networkx as nx

        graph = nx.DiGraph()
        layer_nodes = []
        input_nodes = []
        num_inputs = model.width_in[0]

        for input_idx in range(num_inputs):
            input_name = f"Input_{input_idx}"
            graph.add_node(input_name, layer=0)
            input_nodes.append(input_name)
        layer_nodes.append(input_nodes)

        for layer_idx, layer in enumerate(model.act_fun):
            num_outputs = layer.out_dim
            current_layer_nodes = []
            for node_idx in range(num_outputs):
                node_name = f"L{layer_idx}_N{node_idx}"
                graph.add_node(node_name, layer=layer_idx + 1)
                current_layer_nodes.append(node_name)
            prev_layer_nodes = layer_nodes[-1]
            for prev_node_idx, prev_node in enumerate(prev_layer_nodes):
                for curr_node_idx, curr_node in enumerate(current_layer_nodes):
                    if not torch.all(layer.coef[prev_node_idx, curr_node_idx] == 0):
                        graph.add_edge(prev_node, curr_node)
            layer_nodes.append(current_layer_nodes)

        output_nodes = []
        num_outputs = model.width_out[-1]
        for output_idx in range(num_outputs):
            output_name = f"Output_{output_idx}"
            graph.add_node(output_name, layer=len(layer_nodes))
            output_nodes.append(output_name)
        last_hidden_nodes = layer_nodes[-1]
        for hidden_node_idx, hidden_node in enumerate(last_hidden_nodes):
            for output_idx, output_node in enumerate(output_nodes):
                graph.add_edge(hidden_node, output_node)

        output_nodes = last_hidden_nodes
        for output_node in output_nodes:
            paths = []
            for input_node in input_nodes:
                if nx.has_path(graph, input_node, output_node):
                    # Calcola e salva il percorso
                    path = nx.shortest_path(graph, source=input_node, target=output_node)
                    paths.append(path)
            if not paths:
                raise Exception(f"No continuous path from any input to output {output_node}!")
            else:
                # print(f"Paths to {output_node}: {paths}")
                pass

        for input_node in input_nodes:
            if all(nx.has_path(graph, input_node, output_node) for output_node in output_nodes):
                # print(f"Input {input_node} has a path to all outputs.")
                return True  # Almeno un input raggiunge tutti gli output

        raise Exception("No input has a continuous path to all outputs!")

    def get_kan_graph(model):
        """
        Saves the KAN architecture graph, except for connections whose splines have every coeff set to 0
        and 'dead end' connections
        The label of each spline is reported on the respective connection.
        :param model: KAN model
        :type model: MultKAN
        """
        graph = nx.DiGraph()
        layer_nodes = []
        input_nodes = []
        num_inputs = model.width_in[0]
        for input_idx in range(num_inputs):
            input_name = f"IN{input_idx}"
            graph.add_node(input_name, layer=0)
            input_nodes.append(input_name)
        layer_nodes.append(input_nodes)
        edge_labels = {}
        for layer_idx, layer in enumerate(model.act_fun):
            num_outputs = layer.out_dim
            current_layer_nodes = []
            for node_idx in range(num_outputs):
                node_name = f"L{layer_idx}N{node_idx}"
                graph.add_node(node_name, layer=layer_idx + 1)
                current_layer_nodes.append(node_name)
            prev_layer_nodes = layer_nodes[-1]
            for prev_node_idx, prev_node in enumerate(prev_layer_nodes):
                for curr_node_idx, curr_node in enumerate(current_layer_nodes):
                    if hasattr(layer, 'coef') and not torch.all(layer.coef[prev_node_idx, curr_node_idx] == 0):
                        edge_id = f"{prev_node}-{curr_node}"
                        graph.add_edge(prev_node, curr_node)
                        edge_labels[(prev_node, curr_node)] = edge_id
            layer_nodes.append(current_layer_nodes)
        # check for 'dead end' connections and remove them
        nodes_no_incoming = [node for node in graph.nodes if graph.in_degree(node) == 0]
        nodes_no_outgoing = [node for node in graph.nodes if graph.out_degree(node) == 0]
        input_nodes = [f"IN{i}" for i in range(num_inputs)]
        output_nodes = [node for node, data in graph.nodes(data=True) if
                        data["layer"] == max(data["layer"] for _, data in graph.nodes(data=True))]
        filtered_no_incoming = [node for node in nodes_no_incoming if node not in input_nodes + output_nodes]
        filtered_no_outgoing = [node for node in nodes_no_outgoing if node not in input_nodes + output_nodes]
        connections_to_exclude = []
        for edge in graph.edges:
            source, target = edge
            if target in filtered_no_outgoing:
                connections_to_exclude.append(edge)
            if source in filtered_no_incoming:
                connections_to_exclude.append(edge)

        for connection in connections_to_exclude:
            # print(f'Excluded connection: {connection}')
            pass

        graph.remove_edges_from(connections_to_exclude)
        for connection in connections_to_exclude:
            if connection in edge_labels:
                del edge_labels[connection]
        pos = nx.multipartite_layout(graph, subset_key="layer")
        for layer, nodes in enumerate(layer_nodes):
            total_nodes = len(nodes)
            if total_nodes > 0:
                start_y = -(total_nodes - 1) / 2.0
                for i, node in enumerate(
                        sorted(nodes, key=lambda x: int(x.split('N')[-1]) if 'N' in x else int(x.split('_')[-1]))):
                    pos[node][1] = start_y + i
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=1000)
        nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=40)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", font_size=8, label_pos=0.5)
        plt.title("KAN architecture")
        plt.axis("off")
        fig_name = "kan_arch.png"
        fig_path = os.path.join(results_path, fig_name)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    def function_form(splines):
        """
        Fits KAN splines in different ways.

        :param splines: KAN splines.
        :type splines: dict

        :return: Fitting functions with their coeffs.
        :type: dict
        """
        fit_path = os.path.join(results_path, "fitted_splines")
        if os.path.exists(fit_path):
            shutil.rmtree(fit_path, ignore_errors=True)  # delete al previous fitted spline figures
        os.makedirs(fit_path, exist_ok=True)
        eq_path = os.path.join(fit_path, "equations")
        if os.path.exists(eq_path):
            shutil.rmtree(eq_path, ignore_errors=True)  # delete al previous fitting equations
        os.makedirs(eq_path, exist_ok=True)
        rmses_path = os.path.join(fit_path, "rmse")
        if os.path.exists(rmses_path):
            shutil.rmtree(rmses_path, ignore_errors=True)  # delete al previous fitting rmse
        os.makedirs(rmses_path, exist_ok=True)

        def exponential(x, a, b):
            return a * np.exp(b * x)

        def logarithmic(x, a, b):
            return a * np.log(np.clip(b * x + 1, 1e-9, None))  # Evita valori <= 0

        def polynomial(x, a, b, c, d, e, f, g):
            return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g

        def sine(x, a, b, c):
            return a * np.sin(b * x + c)

        def cosine(x, a, b, c):
            return a * np.cos(b * x + c)

        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        def reciprocal(x, a, b):
            return a / (np.clip(b * x, 1e-9, None))

        def hyperbolic_tangent(x, a, b, c):
            return a * np.tanh(b * x + c)

        def arcsine(x, a, b, c):
            return a * np.arcsin(np.clip(b * x + c, -1, 1))

        def arccosine(x, a, b, c):
            return a * np.arccos(np.clip(b * x + c, -1, 1))

        def sigmoid(x, a, b, c):
            return a / (1 + np.exp(-b * (x - c)))


        # Prepare function forms for fitting
        forms = {
            "Exponential": exponential,
            "Logarithmic": logarithmic,
            "Polynomial": polynomial,
            "Sine": sine,
            "Cosine": cosine,
            "Logistic": logistic,
            "Reciprocal": reciprocal,
            "HyperbolicTangent": hyperbolic_tangent,
            "ArcSine": arcsine,
            "ArcCosine": arccosine,
            "Sigmoid": sigmoid
        }
        # Fit splines with the best function forms
        fitting_functions = {}
        equations = {}
        rmses = {}
        for spline_id, points in splines.items():
            x = np.linspace(-1, 1, len(points))
            y = np.array(points)
            best_fit = None
            best_params = None
            best_error = float("inf")
            for name, func in forms.items():
                try:
                    params, pcov = curve_fit(func, x, y, maxfev=10000)
                    fitted_y = func(x, *params)
                    error = np.sqrt(np.mean((y - fitted_y) ** 2))
                    if np.isfinite(error) and error < best_error:
                        best_fit = name
                        best_params = params
                        best_error = error
                        rmses[spline_id] = round(best_error, 3)
                except Exception as e:
                    # print(f'Bad fitting for {spline_id} with {name}: {e}!')
                    continue
            # Plot and save spline and its fitting function
            plt.figure()
            plt.plot(x, y, label="Original Spline", color="blue")
            if best_fit is not None:
                fitted_y = forms[best_fit](x, *best_params)
                equation = f"{best_fit}: "
                if best_fit == "Exponential":
                    equation += f"{best_params[0]:.2f} * exp({best_params[1]:.2f} * x)"
                elif best_fit == "Logarithmic":
                    equation += f"{best_params[0]:.2f} * log({best_params[1]:.2f} * x + 1)"
                elif best_fit == "Polynomial":
                    equation += f" + ".join([f"{p:.2f} * x^{i}" for i, p in enumerate(reversed(best_params))])
                plt.plot(x, fitted_y, label=f"Fitting func.", color="red")
                plt.title(f"Spline {spline_id} - Best fit: {best_fit} - RMSE: {round(best_error, 3)}", fontsize=10)
                fitting_functions[spline_id] = (forms[best_fit], best_params)
                equations[f'{spline_id}'] = equation
            else:
                plt.title(f"Spline {spline_id} - No Fit Found", fontsize=10)
                pass
            file_name = f'fit_equations.json'
            path = os.path.join(eq_path, file_name)
            with open(path, 'w') as file:
                json.dump(equations, file, indent=2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            fig_name = f"fitted_{spline_id}.png"
            fig_path = os.path.join(fit_path, fig_name)
            plt.savefig(fig_path, dpi=300)
            plt.close()
        rmse_sum = 0
        for value in rmses.values():
            rmse_sum += value
        avg_rmse = round(rmse_sum / (len(rmses.keys())))
        file_name = f'rmses.json'
        path = os.path.join(rmses_path, file_name)
        with open(path, 'w') as file:
            json.dump(rmses, file, indent=2)
        file_name = f'avg_rmse.json'
        path = os.path.join(rmses_path, file_name)
        with open(path, 'w') as file:
            json.dump(avg_rmse, file, indent=2)

        return fitting_functions

    def compose_splines(fitting_functions, input_data, outputs):
        """
        Evaluates the output by applying fitting functions accordind to the Kolmogorov-Arnold Th.

        :param fitting_functions: Fitting functions with their coeffs.
        :type fitting_functions: dict

        :return composed_output, node_outputs: The output and the single-node outputs.
        :rtype: tuple
        """
        def evaluate_function(func, params, x):
            """
            Evaluates a function form.

            :param func: The function to evaluate.
            :type func: function
            :param params: Parameters of the function.
            :type params: list
            :param x: The input data.
            :type x: np.ndarray

            :return: Function value.
            :rtype: float
            """
            # Keep splines abscissa within the domain
            if x > 1:
                x = 1
            elif x < -1:
                x = -1

            return func(x, *params)

        node_outputs = {f"IN{i}": input_data[i] for i in range(len(input_data))}
        connections = sorted(fitting_functions.keys())
        layers = {}
        for conn in connections:
            src, dest = conn.split("-")
            layers.setdefault(dest, []).append(src)
        for dest, sources in layers.items():
            node_output = 0
            for src in sources:
                spline_id = f"{src}-{dest}"
                if spline_id in fitting_functions:
                    func, params = fitting_functions[spline_id]
                    try:
                        input_value = node_outputs[src]
                        node_output += evaluate_function(func, params, input_value)
                    except Exception:
                        # print(f'{src} is a dead node!')
                        continue
            node_outputs[dest] = node_output

        def sort_key(key):
            if key.startswith('IN'):
                return (0, int(key[2:]))
            elif key.startswith('L'):
                layer, node = key[1:].split('N')
                return (int(layer) + 1, int(node))

        node_outputs = dict(sorted(node_outputs.items(), key=lambda x: sort_key(x[0])))
        composed_output = [node_outputs[key] for key in list(node_outputs.keys())[-outputs:]]

        return composed_output, node_outputs

    def compare_node_sums(node_sums, node_sums_parse):
        """
        Calculate differences in node sums (node values) between the KAN and the composed splines
        calculation.

        :param node_sums: The KAN model node values.
        :type node_sums: dict
        :param node_sums_parse: The composed splines node values.
        :type node_sums_parse: dict

        :return comparison: Differences in node values.
        :rtype: dict
        """
        comparison = {}
        for key in node_sums.keys():
            for key_1 in node_sums_parse.keys():
                if key == key_1:
                    original = node_sums.get(key, [0])[0]
                    parsed = node_sums_parse.get(key_1)
                    difference = abs(original - parsed)
                    comparison[key] = [difference]
        return comparison

    def get_kan_splines(model, num_points=100, layer_idx=None):
        """
        Gets KAN splines

        :param model: The KAN model.
        :type model: MultKAN.
        :param num_points: The number of points to extract.
        :type num_points: int
        :param layer_idx: Layer ID.
        :type layer_idx: int

        :return splines: The KAN spline points.
        :rtype: dict
        """
        splines = {}
        x = torch.linspace(-1, 1, steps=num_points)

        org_path = os.path.join(results_path, "original_splines")
        if os.path.exists(org_path):
            shutil.rmtree(org_path, ignore_errors=True)  # delete al previous original spline figures
        os.makedirs(org_path, exist_ok=True)

        if layer_idx is not None:
            layers = [(layer_idx, model.act_fun[layer_idx])]
        else:
            layers = enumerate(model.act_fun)

        for l_idx, layer in layers:
            grid = layer.grid.detach().cpu().numpy()  # Layer grid
            coef = layer.coef.detach().cpu()  # Spline coeffs

            if coef.ndim == 2:
                coef = coef.unsqueeze(0)

            for i in range(layer.in_dim):
                plt.figure(figsize=(10, 6))
                for j in range(layer.out_dim):
                    if l_idx == 0:
                        conn_name = f"IN{i}-L{l_idx}N{j}"
                    else:
                        conn_name = f"L{l_idx - 1}N{i}-L{l_idx}N{j}"
                    try:
                        y = coef2curve(x.unsqueeze(1), torch.tensor(grid), coef, layer.k).detach().cpu().numpy()
                        y_spline = y[:, i, j]
                        # y_spline = [round(x, 3) for x in y_spline]
                        # if not all(c == 0 for c in y_spline):
                        #     splines[conn_name] = y_spline
                        coeff_spline = coef[i, j].detach().cpu().numpy()
                        if not all(c == 0 for c in coeff_spline):
                            splines[conn_name] = y_spline
                            # print(f'Accepted spline: {conn_name}, coeffs: {coeff_spline}')
                        # Plot spline
                        plt.plot(x, y_spline, label=conn_name)
                        plt.title(f"Original spline {conn_name}")
                        plt.xlabel("x")
                        plt.ylabel("y")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        fig_name = f"original_{conn_name}.png"
                        save_path = os.path.join(org_path, fig_name)
                        plt.savefig(save_path, dpi=300)
                        plt.close()

                    except Exception as e:
                        print(f"Error during spline {conn_name} calculation: {e}")

        return splines

    print('*** Simulation START! ***')

    # Clean the plots panel of eventual previous plots
    plt.close('all')

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Results path
    current_folder = os.getcwd()
    rooth_path = os.path.join(current_folder, "results")
    if not os.path.exists(rooth_path):
        os.makedirs(rooth_path, exist_ok=True)
    results_path = os.path.join(rooth_path, f"{test_name}_results")
    if os.path.exists(results_path):
        shutil.rmtree(results_path, ignore_errors=True)  # delete al previous data
    os.makedirs(results_path, exist_ok=True)

    # Set the number of inputs and outputs
    inputs = n_ins
    outputs = n_outs

    # Create the KAN model from the test individual
    individual = create_ind(arch, arg_gr, arg_k, arg_pr_thr)
    kan_model = kan_decode(individual)

    # Check the model continuity before evaluation
    get_kan_graph(kan_model)  # returns dead connections
    check_continuity(kan_model)

    # Evaluate the KAN model and the composed splines one, with random inputs
    input_vector = torch.rand((1, inputs))
    start = time.time()
    output, node_sums = test_kan(kan_model, input_vector)
    elapsed_kan = round(time.time() - start, 3)
    sigmoid_output = torch.sigmoid(output)
    input_data = input_vector.squeeze(0).tolist()
    fitting_functions = function_form(get_kan_splines(kan_model))
    start = time.time()
    output_parse, node_sums_parse = compose_splines(fitting_functions, input_data, outputs)
    elapsed_composed = round(time.time() - start, 3)
    sigmoid_output_parse = torch.sigmoid(torch.tensor(output_parse))

    # Calculate differences between KAN model and composed splines model nodes
    diffs = compare_node_sums(node_sums, node_sums_parse)

    # Prepare results
    kan_output = [[round(val.item(), 3) for val in row] for row in output][0]
    composed_output = [round(float(val), 3) for val in output_parse]
    kan_output_sig = [[round(val.item(), 3) for val in row] for row in sigmoid_output][0]
    composed_output_sig = [round(float(val), 3) for val in sigmoid_output_parse.tolist()]
    composed_error = []
    for i in range(len(kan_output)):
        e = round(abs(composed_output[i] - kan_output[i])/(kan_output[i]+0.001), 2)*100
        composed_error.append(e)
    composed_error_sig = []
    for i in range(len(kan_output_sig)):
        e =  round(abs(composed_output_sig[i] - kan_output_sig[i])/(kan_output_sig[i]+0.001), 2)*100
        composed_error_sig.append(e)
    kan_node_sums = node_sums
    composed_node_sums = node_sums_parse
    node_diffs = diffs
    saved_time = round(100 * (elapsed_kan - elapsed_composed) / elapsed_kan, 2)

    # Save results
    results = {
        "composed_error_sig [%]": composed_error_sig,
        "saved_time [%]": saved_time,
        "composed_time": elapsed_composed,
        "kan_time": elapsed_kan,
        "composed_error [%]": composed_error,
        "kan_output": kan_output,
        "composed_output": composed_output,
        "kan_output_sig": kan_output_sig,
        "composed_output_sig": composed_output_sig,
        "kan_node_sums": kan_node_sums,
        "composed_node_sums": composed_node_sums,
        "node_diffs": node_diffs
    }
    file_name = "results.json"
    file_path = os.path.join(results_path, file_name)
    with open(file_path, "w") as file:
        json.dump(results, file, indent=2)

    print(f'The KAN composed model makes the following percentage errors on the '
          f'final output nodes: \n '
          f'{composed_error_sig} \n'
          f'Moreover, the percentage time saved than the original KAN model is: \n'
          f'{saved_time} \n'
          f'\n')

    print('*** Simulation END! ***')
