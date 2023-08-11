import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='EPS',
                    help="dataset from ['EPS']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='USAD',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--use_fl', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--client', required=False, type=int, help="Client ID")
parser.add_argument('--name', required=False, type=str, help="Name of the experiment")
parser.add_argument(
    "--server_address",
    type=str,
    default="127.0.0.1:8080",
    required=False,
    help=f"gRPC server address",
)
parser.add_argument(
    "--cid", 
    default=1,
    type=int, 
    required=False, 
    help="Client CID (no default)"
)
parser.add_argument(
    "--log_host",
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=10,
    help="Number of rounds of federated learning (default: 1)",
)
args = parser.parse_args()