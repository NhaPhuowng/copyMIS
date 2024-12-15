import dimod
import networkx as nx
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import dimod
from ortools.linear_solver import pywraplp
import pandas as pd

# token = "DEV-7affe1a83dbe06fa17c9a260577608396c251455"
# token = "DEV-b28f5c26b9419829978caa8899867ab5c25f9802"

# DEV-898779584c4bed23fcf5bcbd657344d29493c2b9
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
    check_violet_list = []
    for i in data_sample.keys():
        if data_sample.get(i) == 1:
            check_violet_list.append(i)
        
    check_violet_edges = list(combinations(check_violet_list, 2))
    for edge in check_violet_edges:
        if(edge in graph.edges()):
            return 1
            
    return 0

def count_percet_solution(response_data, lowest_energy, graph):
    num_of_correct_solution = 0
    for data in response_data:
        if (data.energy == lowest_energy) and check_violet(data.sample, graph) == 0:
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
    file_to_read = "data_1.txt"  # File cần đọc

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
    # random.seed(1)
    # create_random_graph(40, 312)
    
    input_folder = "input_data"
    output_folder = "SA/Running/Gamma2"
    output_csv = "SA/output_csv"
    
    for i in range (1, 16):
        file_to_read = "data_" + str(i) + ".txt"
        file_path = os.path.join(input_folder, file_to_read)
        #print(file_path)
    
        # Kiểm tra nếu file tồn tại
        if os.path.exists(file_path):
            G = nx.Graph()
            with open(file_path, "r") as file:
                for line in file:
                    u, v = map(int, line.split())  # Đọc các cạnh từ file
                    G.add_edge(u, v)
        # print(G.edges)
    
        # Generate QUBO
        penalty_weigth_num = 2.0
        Q = create_mis_qubo(G, penalty_weight=penalty_weigth_num)
 
        sampler = SimulatedAnnealingSampler()

        response = sampler.sample(Q, num_reads = 1000)
    
        # using ortool
        list_of_ones = [1] * len(G.nodes)
        res_ortools = maximum_weighted_independent_set(list_of_ones, G.edges())
        lowest_energy_orTools = - res_ortools
        #print("Ortool: ", res_ortools)
    
        # print(response['time'])
        print(response.info)
        print("------------")
        print(response)
        print("------------")
        #rint(response[timing])
    
        for data in response.data():
            print(data)
        print("-------------------------------------------")
  
        print("So dinh:", G.number_of_nodes())
        print("So canh:", G.number_of_edges())
        print("Mat do do thi:", count_denity_graph(G.number_of_nodes(), G.number_of_edges()))
        print("Nang luong thap nhat bi sai ban dau la: ", response.first.energy)
        print("So lan vi pham rang buoc: ", count_num_penalty(response.data(), G))
        print("So solution dung la:", count_percet_solution(response.data(), lowest_energy_orTools, G))
        print("Phan tram so cau tra loi dung: ", count_percet_solution(response.data(), lowest_energy_orTools, G)/10)
        print("Best solutions are {}% of samples.".format(len(response.lowest(atol=0.5).record.energy)/1000))
        print("penalty_weigth_num, penalty_weigth_num")
        print(response.info["timing"])
        #print("Nang luong thap nhat theo Exact Solver: ", min_energy_ExactSolver)
        print("Nang luong thap nhat va loi giai toi uu ortools: ", res_ortools)
    
        file_to_write = "SA_gamma_2" + str(i) + ".json"
        file_path_write = os.path.join(output_folder, file_to_write)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Chuẩn bị dữ liệu để ghi vào file JSON
        output_data = []

        # Chuyển đổi dữ liệu từ Sample thành dạng có thể lưu vào JSON
        for data in response.data():
            #sample = data.sample
            energy = data.energy
            num_occurrences = data.num_occurrences
            sample_info = {
            #"sample": data.sample,  # Đưa dictionary sample vào (các giá trị như {0: 0, 1: 1, ...})
            "energy": energy,
            "num_occurrences": int(num_occurrences)
            }
            output_data.append(sample_info)
    
        result_info = {
            "So dinh": G.number_of_nodes(),
            "So canh": G.number_of_edges(),
            "Mat do do thi": count_denity_graph(G.number_of_nodes(), G.number_of_edges()),
            "Nang luong thap nhat bi sai ban dau": response.first.energy,
            "So lan vi pham rang buoc": count_num_penalty(response.data(), G),
            "So solution dung la": int(count_percet_solution(response.data(), lowest_energy_orTools, G)),
            "Phan tram so cau tra loi dung": count_percet_solution(response.data(), lowest_energy_orTools, G)/1000,
            "Best solutions of samples %": format(len(response.lowest(atol=0.5).record.energy)/10),
            "penalty_weigth_num": penalty_weigth_num,
            "Thoi gian chay": response.info["timing"],
            #"Nang luong thap nhat theo ExactSolver": min_energy_ExactSolver,
            "Nang luong thap nhat va cac loi giai toi uu theo ortools": res_ortools
        }
        output_data.append(result_info)

    # Lưu dữ liệu vào file JSON
        try:
            with open(file_path_write, "w") as output_file:
                json.dump(output_data, output_file, indent=4)
            print(f"Results have been saved to {file_path_write}")
        except Exception as e:
            print(f"Error while writing JSON: {e}")
            
        data_for_df.append({ # type: ignore
                "file_read": file_to_read,
                "file_write": file_to_write,
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "graph_density": 2 * len(G.edges) / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
                "gamma": penalty_weigth_num,
                "lowest_energy_dwave": response.first.energy,
                "lowest_energy_or_tools": lowest_energy_orTools,
                "vi pham": count_num_penalty(response.data(), G),
                "correct_solutions": count_percet_solution(response.data(), lowest_energy_orTools, G),
                "percentage_correct": count_percet_solution(response.data(), lowest_energy_orTools, G) / 1000,
                "SA timing": response.info["timing"]
            })
        
        df = pd.DataFrame(data_for_df) # type: ignore
        print(df)
    
        if not os.path.exists(output_csv):
            os.makedirs(output_csv)
            
        df.to_csv(os.path.join(output_csv, "SA_gamma2.csv"), index=False)

        print("Data has been saved to SA_gamma1.csv")