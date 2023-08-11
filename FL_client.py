import argparse
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import result
from src.pot import pot_eval

import flwr as fl
import numpy as np
import torch
from time import time
from utils import *
import timeit
from flwr.common import Status, Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersRes, NDArrays


DEVICE = 'cpu'


def get_weights(model: torch.nn.ModuleList) -> fl.common.NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


class EPSClient(fl.client.Client):
    def __init__(self, 
        cid: str, 
        trainD, 
        testD,
        trainO,
        testO,
        labels,
        model,
        optimizer,
        scheduler,
        epoch,
        accuracy_list
    ) -> None:
        super().__init__()
        self.cid = cid
        self.trainD = trainD
        self.testD = testD
        self.trainO = trainO
        self.testO = testO
        self.labels = labels
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.accuracy_list = accuracy_list

        # Print initial 
        print(f"Client {self.cid}: ____init_____")
    

    def get_parameters(self) -> fl.common.GetParametersRes:
        print(f"Client {self.cid}: get_parameters")
        weights: NDArrays = get_weights(self.model)
        parameters = fl.common.ndarrays_to_parameters(weights)
        return GetParametersRes(parameters=parameters)


    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        epochs = int(config["epochs"])

        set_weights(self.model, weights)

        e = self.epoch + 1
        start = time()
        for e in list(range(self.epoch + 1, self.epoch + epochs + 1)):
            lossT, lr = backprop(e, self.model, self.trainD, self.trainO, self.optimizer, self.scheduler)
            self.accuracy_list.append((lossT, lr))
        print('Training time: ' + "{:10.4f}".format(time() - start) + ' s')

        weights_prime: NDArrays = get_weights(self.model)
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        num_examples_train = len(self.trainD)
        
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            status=Status(
                code=200,
                message="Successfully fit",
            ),
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")
        torch.zero_grad = True
        self.model.eval()

        weights = fl.common.parameters_to_ndarrays(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)

        loss, y_pred = backprop(0, self.model, self.testD, self.testO, self.optimizer, self.scheduler, training=False)
        lossT, _ = backprop(0, self.model, self.trainD, self.trainO, self.optimizer, self.scheduler, training=False)

        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(self.labels, axis=1) >= 1) + 0
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        
        # pprint(result)

        metrics = {"accuracy": float(result['accuracy'])}
        return EvaluateRes(
            status=Status(
                code=200,
                message="Successfully Evaluate",
            ),
            loss=float(np.mean(lossFinal)), num_examples=len(self.testD), metrics=metrics
        )


def main():
    fl.common.logger.configure(f"Client_{args.cid}", host=args.log_host)

    train_loader, test_loader, labels = load_dataset_partition(args.cid)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD

    if model.name in ['USAD']: 
    	trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    client = EPSClient(
        cid=args.cid,
        trainD=trainD,
        testD=testD,
        trainO=trainO,
        testO=testO,
        labels=labels,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        accuracy_list=accuracy_list
    )
    fl.client.start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()