"""Flower server example."""

from typing import List, Tuple
import argparse
import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

parser = argparse.ArgumentParser(description="Server")
parser.add_argument("--port", type=str, required=True,)
args = parser.parse_args()
# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=3,  # Minimum number of clients to be sampled for the next round
    min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start
    #min_eval_clients = 1
)

# Start Flower server
fl.server.start_server(
    server_address=args.port,
    config=fl.server.ServerConfig(num_rounds=101),
    strategy=strategy,
)
