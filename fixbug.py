import dimod
import networkx as nx
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import dimod
import pandas as pd
from ortools.linear_solver import pywraplp

# token = "DEV-7affe1a83dbe06fa17c9a260577608396c251455"
# token = "DEV-b28f5c26b9419829978caa8899867ab5c25f9802"

# DEV-898779584c4bed23fcf5bcbd657344d29493c2b9

#DEV-1b5e467ff8bc1e44f94062b62fc90af26dbe982e
#pip install dwave-ocean-sdk
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from itertools import combinations


def create_mis_qubo(graph, penalty_weight=1.0):
    """
    Create a QUBO formulation for the Maximum Independent Set problem.

    Parameters:
        graph (networkx.Graph): Input graph with nodes and edges.
        penalty_weight (float): Weight for the penalty term.

    Returns:
        dimod.BinaryQuadraticModel: The QUBO for the Maximum Independent Set.
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Add linear terms for maximizing the size of the independent set
    for node in graph.nodes:
        bqm.add_variable(node, -1)  # Coefficient for H1: -x_i

    # Add quadratic terms for penalizing adjacent nodes in the set
    for edge in graph.edges:
        i, j = edge
        bqm.add_interaction(i, j, penalty_weight)  # Coefficient for H2: lambda * x_i * x_j

    return bqm

def count_num_penalty(response_data, graph):
    total_penalty = 0
    for data in response_data:
        res = data.sample
        test_list = []  # khoi tao danh sach cac dinh duoc chon
        penalty = 0  # khoi tao so lan vi pham rang buoc

        # tao list cac dinh duoc chon
        for i in res.keys():
            #print(res.get(i), end=" ")
            if res.get(i) == 1:
                 test_list.append(i)

        # tu list do tao nen cac cap canh kha thi, dem so cap thuoc do thi G
        possible_edge = list(combinations(test_list, 2))
        for x in possible_edge:
            if (x in graph.edges):
                penalty += 1
                total_penalty += 1
    return total_penalty

def check_violet(data_sample, graph):
    #num = 0
    #print(data_sample)
    check_violet_list = []
    for i in data_sample.keys():
        if data_sample.get(i) == 1:
            check_violet_list.append(i)
            #print("Nut ", i, "co gia tri 1")
    check_violet_edges = list(combinations(check_violet_list, 2))
    #print("Danh sach cac cap canh duoc chon", check_violet_edges)
    for edge in check_violet_edges:
        if(edge in graph.edges()):
            return 1
            
    return 0
            
def count_percet_solution(response_data, lowest_energy, graph):
    num_of_correct_solution = 0
    for data in response_data:
        if (data.energy == lowest_energy_orTools) and check_violet(data.sample, graph) == 0:
            print("check_violet(data.sample, graph): ", check_violet(data.sample, graph))
            print("data.num_occurrences: ", data.num_occurrences)
            num_of_correct_solution += data.num_occurrences
    return num_of_correct_solution

def count_denity_graph(n, num_edges):
    return 2 * num_edges / (n * (n - 1))

def create_random_graph(n, num_edges):
    # Tạo danh sách các cạnh trong đồ thị đầy đủ
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    # Chọn ngẫu nhiên 147 cạnh từ danh sách
    random.shuffle(edges)
    selected_edges = edges[:num_edges]
    
    input_folder = "input_data"  # Thư mục chứa các file TXT
    file_to_read = "data_15.txt"  # File cần đọc

    # Đường dẫn đầy đủ đến file
    file_path = os.path.join(input_folder, file_to_read)
    # In các cạnh theo yêu cầu
    if os.path.exists(file_path):
        with open(file_path, "w") as f:
            for edge in selected_edges:
                f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Đồ thị với {n} đỉnh, {num_edges} cạnh, có mật độ {count_denity_graph(n, num_edges)} đã được lưu vào file {file_path}")

    
def maximum_weighted_independent_set(weights, edges):
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None

    num_nodes = len(weights)

    # Decision variables: x[i] = 1 if node i is in the independent set, 0 otherwise
    x = {}
    for i in range(num_nodes):
        x[i] = solver.BoolVar(f'x[{i}]')

    # Objective function: maximize sum of weights * x[i]
    objective = solver.Objective()
    for i in range(num_nodes):
        objective.SetCoefficient(x[i], weights[i])
    objective.SetMaximization()

    # Constraints: For each edge (i, j), ensure that x[i] + x[j] <= 1
    for (i, j) in edges:
        solver.Add(x[i] + x[j] <= 1)

    # Solve the problem
    status = solver.Solve()
    print('TimeRunning %f' % (solver.wall_time() / 1000.0))
    print('Problem solved in %d iterations' % solver.iterations())
    # Check if a solution was found
    if status == pywraplp.Solver.OPTIMAL:
        print('OptimalSolution', solver.Objective().Value())
        independent_set = [i for i in range(num_nodes) if x[i].solution_value() == 1]
        total_weight = sum(weights[i] for i in independent_set)
        return total_weight
    else:
        return 0

data_for_df = []
if __name__ == "__main__":
    
    # create radom graph and save to folder input
    # random.seed(15)
    # create_random_graph(40, 312)
    
    
    input_folder = "input_data"
    output_folder = "NEW/PickAnT/gamma2_AnT200"
    output_csv = "NEW/output_csv"
    
    
    for i in range(1, 2):
        file_to_read = "data_" + str(i) + ".txt"
        file_path = os.path.join(input_folder, file_to_read)
        print(file_path)
        
        if os.path.exists(file_path):
            print(file_path)
            G = nx.Graph()
            with open(file_path, "r") as file:
                for line in file:
                    u, v = map(int, line.split()) 
                    G.add_edge(u, v)
                #print(G.edges)
        
        list_of_ones = [1] * len(G.nodes)
        res_ortools = maximum_weighted_independent_set(list_of_ones, G.edges())
    
        penalty_weigth_num = 1
        Q = create_mis_qubo(G, penalty_weight=penalty_weigth_num)
        #print(Q)
    
        chainstrength = 8
        numruns = 1000
        annealingTime = 200
        sampler = EmbeddingComposite(DWaveSampler(token='DEV-1b5e467ff8bc1e44f94062b62fc90af26dbe982e'))
        response = sampler.sample(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               annealing_time=annealingTime,
                               label='Maximum Independent Set')
    

        response = sampler.sample(Q, num_reads = 1000)
        
        timing_info = response.info["timing"]
        
    

        lowest_energy = response.first.energy
        lowest_energy_orTools = - res_ortools
    
        x = 0
        # for data in response.data():
            # print(data.sample)
        print(count_percet_solution(response.data(), lowest_energy_orTools, G))
        print("Tong so rang buoc vi pham:", count_num_penalty(response.data(), G))
            
        
        # print("------------------------------------------------") 
        # print(response)
        # print("\\\\\\\\\\\\")
        # print(response.first.sample)

        
        
        print("------------------#########")
    
            
    #     print("-------------------------------------------")
    #     print("Gamma:", penalty_weigth_num)
    #     print("Annealing_time:", annealingTime)
    #     print("So dinh:", G.number_of_nodes())
    #     print("So canh:", G.number_of_edges())
    #     print("Mat do do thi:", count_denity_graph(G.number_of_nodes(), G.number_of_edges()))
    #     print("Nang luong thap nhat bi sai ban dau la: ", lowest_energy)
    #     print("So lan vi pham rang buoc: ", count_num_penalty(response.data(), G))
    #     print("So solution dung la:", count_percet_solution(response.data(), lowest_energy_orTools, G))
    #     print("Phan tram so cau tra loi dung: ", count_percet_solution(response.data(), lowest_energy_orTools, G)/10)
    #     print("Best solutions are {}% of samples.".format(len(response.lowest(atol=0.5).record.energy)/10))
    #     print(response.info["timing"])
    #     print("Nang luong thap nhat va loi giai toi uu ortools: ", res_ortools)
    #     print("qpu_anneal_time_per_sample:", timing_info['qpu_anneal_time_per_sample'])
    #     print("qpu_access_time:", timing_info['qpu_access_time'])
        


    #     # Save data to folder out_put
    #     file_to_write = "QA_gamma2_anT200" + str(i) +".json"
    #     file_path_write = os.path.join(output_folder, file_to_write)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)

    #     output_data = []

    #     for data in response.data():
    #         energy = data.energy
    #         num_occurrences = data.num_occurrences
    #         chain_break_fraction = data.chain_break_fraction
    #         sample_info = {
    #         "energy": energy,
    #         "num_occurrences": int(num_occurrences),
    #         "chain_break_fraction": chain_break_fraction
    #         }
    #         output_data.append(sample_info)
    
    #     result_info = {
    #         "Gamma": penalty_weigth_num,
    #         "Annealing_time": annealingTime,
    #         "So dinh": G.number_of_nodes(),
    #         "So canh": G.number_of_edges(),
    #         "Mat do do thi": count_denity_graph(G.number_of_nodes(), G.number_of_edges()),
    #         "Nang luong thap nhat bi sai ban dau": lowest_energy,
    #         "vi pham": count_num_penalty(response.data(), G),
    #         "solution dung": int(count_percet_solution(response.data(), lowest_energy_orTools, G)),
    #         "Percent solution dung": count_percet_solution(response.data(), lowest_energy_orTools, G)/1000,
    #         "Best solutions of samples %": format(len(response.lowest(atol=0.5).record.energy)/10),
    #         "Thoi gian chay": response.info["timing"],
    #         "MIS Ortools": res_ortools
    #     }
    #     output_data.append(result_info)

    #     try:
    #         with open(file_path_write, "w") as output_file:
    #             json.dump(output_data, output_file, indent=4)
    #         print(f"Results have been saved to {file_path_write}")
    #     except Exception as e:
    #         print(f"Error while writing JSON: {e}")
        
        
    #     #to Data Frame
    #     data_for_df.append({ # type: ignore
    #             "file_read": file_to_read,
    #             "file_write": file_to_write,
    #             "num_nodes": G.number_of_nodes(),
    #             "num_edges": G.number_of_edges(),
    #             "graph_density": 2 * len(G.edges) / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
    #             "gamma": penalty_weigth_num,
    #             "annealing_time": annealingTime,
    #             "lowest_energy_dwave": lowest_energy,
    #             "lowest_energy_or_tools": lowest_energy_orTools,
    #             "vi pham": count_num_penalty(response.data(), G),
    #             "correct_solutions": count_percet_solution(response.data(), lowest_energy_orTools),
    #             "percentage_correct": count_percet_solution(response.data(), lowest_energy_orTools) / 10,
    #             "qpu_anneal_time_per_sample": timing_info['qpu_anneal_time_per_sample'],
    #             "qpu_access_time": timing_info['qpu_access_time']
                
    #         })
        
    # df = pd.DataFrame(data_for_df) # type: ignore
    # print(df)
    
    # if not os.path.exists(output_csv):
    #         os.makedirs(output_csv)
            
    # df.to_csv(os.path.join(output_csv, "QA_gamma2_anT200.csv"), index=False)

    # print("Data has been saved to gamma0_5_AnTime50_again.csv")

                

   