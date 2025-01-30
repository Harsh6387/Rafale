# -*- coding: utf-8 -*-


import flwr as fl
from typing import List, Tuple, Optional, Dict, Union
import numpy as np  # Make sure numpy is imported
import flwr as fl
from flwr.server.client_proxy import ClientProxy  # Import ClientProxy
from flwr.common import FitRes, Parameters, Scalar   # Ensure Flower (flwr) is imported correctly



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights_student.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,

)
fl.server.start_server( server_address="0.0.0.0:8082",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,)