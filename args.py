import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training for STR, DNW and GMP")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-min",
        default=0.01,
        type=float,
        metavar="LR-min",
        help="lowest possible lr"    
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--init_prune_epoch", default=0, type=int, help="Init epoch for pruning in GMP"
    )
    parser.add_argument(
        "--final_prune_epoch", default=100, type=int, help="Final epoch for pruning in GMP"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--num-classes",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--random-mask",
        action="store_true",
        help="Whether or not to use a random mask when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only mask (this freezes weights)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--dense-conv-model", action="store_true", default=False, help="Store a model variant of the given pretrained model that is compatible to CNNs with DenseConv (nn.Conv2d)"
    )
    parser.add_argument(
        "--st-decay", type=float, default=None, help="decay for sparse thresh. If none then use normal weight decay."
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )

    parser.add_argument(
        "--sInit-type",
        type=str,
        help="type of sInit",
        default="constant",
    )
    
    parser.add_argument(
        "--sInit-value",
        type=float,
        help="initial value for sInit",
        default=100,
    )

    parser.add_argument(
        "--sparse-function", type=str, default='sigmoid', help="choice of g(s)"
    )

    parser.add_argument(
        "--er-sparse-init", type=float, default=0.5, help="initial density of ER network"
    )

    parser.add_argument(
        "--er-sparse-method", type=str, default='uniform', help="layerwise ratios for ER mask"
    )

    parser.add_argument(
        "--er-sparsity-file", type=str, help="load sparsity of ER mask from a json file which has layerwise sparsity ratios saved"
    )

    parser.add_argument(
        "--use-budget", action="store_true", help="use the budget from the pretrained model."
    )
    parser.add_argument(
        "--ignore-pretrained-weights", action="store_true", help="ignore the weights of a pretrained model."
    )
    
    
    parser.add_argument("--threshold-list", nargs="*", type=float, default=[0.5])
    parser.add_argument("--warmup", action='store_true', default=True)
    parser.add_argument("--reset-weights", action='store_true', default=False)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--pruner", type=str, default='', help="pruning method")
    parser.add_argument("--prune-random-structured", action='store_true', default=False)
    parser.add_argument("--structured-pruned-iters", type=int, default=10)
    parser.add_argument("--structured-budget", type=int, default=100)
    parser.add_argument("--fix-first-last-structured", action='store_true', default=False)

    parser.add_argument("--result-dir", type=str, default='results', help="pruning method")
    parser.add_argument("--expt-setup", type=str, default='cispa', help="which cluster we run the experiment")
    parser.add_argument("--resnet-type", type=str, default='small-dense', help="resnet-width")
    parser.add_argument("--width", type=int, default=64, help="width resnet small dense")

    parser.add_argument("--structured-prune-type", type=str, default='balanced', help="pruning method for the structured part, used in main_combined")
    parser.add_argument("--structured-prune-ratio", type=float, default=0.005, help = 'balanced removal of neurons, 0.01 for ResNet18 full width')
    parser.add_argument("--target-width", type=int, default=256, help = 'target width to achieve after structure pruning in each layer')
    parser.add_argument("--hessian", action='store_true', default=False)

    # These attributes are only for imagenet (used in main_imagenet)
    parser.add_argument("--warmup-lr", type=float, default=0.01)
    parser.add_argument("--prune-lr", type=float, default=0.01)

    parser.add_argument("--mnist-width", type=int, default=256)

    # 
    parser.add_argument("--target-expt-name", type=str, default='', help="target mask experiment name")

    #
    parser.add_argument("--load-before-prune", action='store_true', default=False)
    parser.add_argument("--load-model-name", type=str, default='load model from checkpoint', help="target mask to prune to")
    parser.add_argument("--load-mask-name", type=str, default='', help="target mask, er model init")

    parser.add_argument("--prune-scheduler", type=str, default='cosine', help="which scheduler to use for pruning")

    parser.add_argument("--constant-prune-lr", type=float, default=0.1, help="which scheduler to use for pruning")

    parser.add_argument("--lbfgs-compensate", action='store_true', default=False)
    parser.add_argument("--reset-weight-sign", action='store_true', default=False)
    parser.add_argument("--load-only-model", action='store_true', default=False)
    parser.add_argument("--load-only-mask", action='store_true', default=False)
    parser.add_argument("--load-model-mask-and-sign", action='store_true', default=False)
    # load_shuffled_model_mask_and_sign
    parser.add_argument("--load-shuffled-model-mask-and-sign", action='store_true', default=False)

    parser.add_argument("--reset-weight-amplitude", action='store_true', default=False)
    parser.add_argument("--reset-shuffled-amplitude", action='store_true', default=False)

    parser.add_argument("--reset-only-weight-sign", action='store_true', default=False)
    parser.add_argument("--reset-only-bn", action='store_true', default=False)
    parser.add_argument("--reset-only-bn-amplitude", action='store_true', default=False)
    parser.add_argument("--integrate-bn", action='store_true', default=False)
    parser.add_argument("--reset-only-weights-not-bn", action='store_true', default=False)
    parser.add_argument("--train-only-bn", action='store_true', default=False)
    parser.add_argument("--train-only-bn-and-linear", action='store_true', default=False)

    # train_only_bn_and_linear
    # reset_only_weights_not_bn
    parser.add_argument("--bn-prune-threshold", type=float, default=0.01)
# reset_shuffled_amplitude
# reset_only_bn_amplitude
    parser.add_argument("--load-expt-sign", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    
    # reset_only_weight_sign
    # for linear mode connectivity

    parser.add_argument("--linear-mode-model-1", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    parser.add_argument("--linear-mode-model-2", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    parser.add_argument("--linear-mode-mask-1", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    parser.add_argument("--linear-mode-mask-2", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    
    # adding knowledge distillation parameters alpha and temperature

    parser.add_argument("--kd", action='store_true', default=False)
    parser.add_argument("--alpha-kd", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=6)
    parser.add_argument("--teacher-model", type=str, default='load teacher model from checkpoint', help="teacher to train with")
    parser.add_argument("--teacher-mask", type=str, default='load teacher model from checkpoint', help="teacher to train with")

    # Resuming pruning-training from a given level
    parser.add_argument("--resume-training-from-level", action='store_true', default=False)
    parser.add_argument("--resume-level", type=int, default=0)

    parser.add_argument("--bn-layer-idx", type=int, default=5, help='the layer index of which you want to track the stats')
    parser.add_argument("--track-bn-running-stats", action='store_true', default=False)

    parser.add_argument("--plot-activation-label", nargs="*", type=float, default=[0], help='Target Label of the Activation to be plotted')

    parser.add_argument("--track-activation", action='store_true', default=False)

    # To check how the initial signs affect generalization
    parser.add_argument("--fix-sign-and-train", action='store_true', default=False)
    parser.add_argument("--perturb-signs", action='store_true', default=False)
    parser.add_argument("--perturb-sign-ratio", type=float, default=0.3)
    parser.add_argument("--perturb-sign-every", type=int, default=30, help='perturb sign after every - epochs')

    # This flag is only relevant for grokking experiments
    parser.add_argument("--load-and-train", action='store_true', default=False)

    # parser.add_argument("--load-expt-sign", type=str, default='', help='model whose sign to add to weights during reset of IMP')
    # algorithmic_set
    parser.add_argument("--algorithmic-set", type=str, default='sum', help='type of algorithic problem to use for grokking experiments, sum, difference, product')
    

    args = parser.parse_args()

    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
