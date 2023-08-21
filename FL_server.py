import argparse
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import result
from src.pot import pot_eval
from src.plotting import plotter

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersRes, NDArrays
import numpy as np
import torch
from time import time

from utils import *


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(20),
        "model": args.model,
        "dataset": args.dataset,
    }
    return config
        

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(
    model, optimizer, scheduler, trainD, testD, trainO, testO, labels
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        torch.zero_grad = True
        model.eval()

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        cost_in_byte = calculate_model_weights(model)

        print(f"Round {server_round} | Cost in bytes: {cost_in_byte}")
        
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        plotter(f'USAD_FL', testO, y_pred, loss, labels)
        lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)

        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        pprint(result)

        return np.mean(lossFinal), {"accuracy": result['accuracy']}
    return evaluate       


def calculate_model_weights(model):
    total_size = 0
    for param in model.parameters():
        param_size = param.element_size() * param.nelement()
        total_size += param_size
    return total_size


class SaveModelStrategy(fl.server.strategy.FedAdagrad):
    def aggregate_fit(self, 
        server_round: int, 
        results: List[Tuple[fl.common.FitRes, Metrics]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[fl.common.FitRes, Metrics]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        os.makedirs(f"checkpoints/USAD_EPS_FL", exist_ok=True)
        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy_list': accuracy_list}, f"checkpoints/USAD_EPS_FL/model_round_{server_round}.ckpt")
        return aggregated_parameters, aggregated_metrics


def main() -> None:
    print(args)

    global model
    global epoch
    global optimizer
    global scheduler
    global accuracy_list
    
    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server")
    
    # Load evaluation data
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    model_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=args.min_sample_size,
        min_evaluate_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        evaluate_fn=get_evaluate_fn(model, optimizer, scheduler, trainD, testD, trainO, testO, labels),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(model_weights),
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    history = fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=10)
    )


if __name__ == "__main__":
    main()