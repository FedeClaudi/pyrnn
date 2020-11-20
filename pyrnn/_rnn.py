import torch
import torch.nn as nn
from rich import print
from rich.prompt import Confirm
from myterial import amber_light, orange
import numpy as np
from pathlib import Path


from ._progress import base_progress
from ._utils import npify
from ._io import load_json, save_json
from ._trainer import Trainer


# --------------------------- Placeholder functions -------------------------- #


def on_epoch_start(rnn, *args, **kwargs):
    """
    Called at start of each epoch during training.
    """
    return


def on_batch_start(rnn, X, Y):
    """
    Called at start of each batch during training.
    Can return an initialized hidden state h.
    """
    return


# ---------------------------- Weights initializer --------------------------- #


class RecurrentWeightsInitializer(object):
    """
    This class implements biological constraints
    on the recurrent weights of a RNN subclassing RNNBase
    """

    def __init__(
        self,
        initial_weights,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
    ):
        """
        This class implements biological constraints
        on the recurrent weights of a RNN subclassing RNNBase.
        Updated weights can be accessed at RecurrentWeightsInitializer.weights.

        Arguments:
            initial_weights (np.ndarray): (n_units * n_units) np.array
                with recurrent weights initialized by pythorch.
            dale_ratio (float): of not None should be a float in range (0, 1)
                specifying which proportion of units should be excitatory
            autopses (bool): if False autopses are removed from weights
                (diagonal elements on weights matrix)
            connectivity (): not yet implemented: will be used to contraint
                connectivity among unnits
        """
        self.weights = initial_weights
        self.connectivity = np.ones_like(self.weights)

        if connectivity is not None:
            self._set_connectivity(connectivity)

        if dale_ratio is not None:
            self._apply_dale_ratio(dale_ratio)

        if not autopses:
            self._remove_autopses()

    def _remove_autopses(self):
        """
        Removes diagonal elements form weights matrix
        """
        np.fill_diagonal(self.weights, 0)
        return

    def _apply_dale_ratio(self, dale_ratio):
        """
        Implements the dale ratio to specify
        the proportion of excitatory/inhibitory units.

        Arguments:
            dale_ratio (float): in range (0, 1). Proportion of excitatory
                units
        """
        if dale_ratio < 0 or dale_ratio > 1:
            raise ValueError(f"Invalid dale ratio value of: {dale_ratio}")

        n_units = len(self.weights)
        n_excitatory = int(np.floor(n_units * dale_ratio))

        dale_vec = np.ones(n_units)
        dale_vec[n_excitatory:] = -1

        self.weights = np.matmul(np.abs(self.weights), np.diag(dale_vec))

    def _set_connectivity(self, connectivity):
        # see: https://colab.research.google.com/github/murraylab/PsychRNN/blob/master/docs/notebooks/BiologicalConstraints.ipynb
        # and: https://github.com/murraylab/PsychRNN/blob/master/psychrnn/backend/initializations.py
        raise NotImplementedError(
            "Need to set up a method to enforce connectivity constraints"
        )


# ------------------------------ Base RNN class ------------------------------ #


class RNNBase(nn.Module, Trainer):
    """
    Base RNN class, implements method
    to save/load RNNs, apply constraints on
    recurrent weights and train/predict with the network.
    This class if not a functinal RNN by itself, but should
    be subclasses by RNN classes.

    """

    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
        on_gpu=False,
    ):
        """
        Base RNN class, implements method
        to save/load RNNs, apply constraints on
        recurrent weights and train/predict with the network.
        This class if not a functinal RNN by itself, but should
        be subclasses by RNN classes.

        Arguments:
            input_size (int): number of inputs
            output_size (int): number of outputs
            n_units (int): number of units in RNN layer
            dale_ratio (float): percentage of excitatory units.
                Use None to ignore
            autopses (bool): pass False to remove autopses from
                recurrent weights
            connectivity (): not implemented yet
            on_gpu (bool): if true computation is carried out on a GPU
        """
        super(RNNBase, self).__init__()
        Trainer.__init__(self)

        self.on_gpu = on_gpu

        self.n_units = n_units
        self.output_size = output_size
        self.input_size = input_size

        self.dale_ratio = dale_ratio
        self.autopses = autopses
        self.connectivity = connectivity

        # recurrent weights are not definite yet
        self._is_built = False
        self.params = {}

        # place holder functions, to be replaced if necessary
        self.on_epoch_start = on_epoch_start
        self.on_batch_start = on_batch_start

    @classmethod
    def from_dict(cls, params):
        """
        Create a class instance from a dictionary of arguments

        Arguments:
            params (dict): dictionary of params to be passed as
                keyword arguments to the RNN class
        """
        try:
            rnn = cls(**params)
        except Exception as e:
            raise ValueError(f"Failed to isntantiate RNN from dictionary: {e}")
        return rnn

    @classmethod
    def from_json(cls, filepath):
        """
        Create a class instance by loading a parameters json file

        Arguments:
            filepath (str, Path): path to a .json file with a
                RNN parameter's
        """
        return cls.from_dict(load_json(filepath))

    def save_params(self, filepath):
        """
        Save the network's parameter's to a json file

        Arguments:
            filepath (str, Path); path to a .json file
        """
        save_json(filepath, self.params)

    def save(self, path):
        """
        Save model to .pt file
        """
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")
        path = Path(path)
        if path.exists():
            if not Confirm.ask(
                f"{path.name} exists already, overwrite?", default=True
            ):
                print("Okay, not saving anything then")
                return
        print(f"[{amber_light}]Saving model at: [{orange}]{path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        """
        Load model from .pt file
        """
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")

        print(f"[{amber_light}]Loading model from: [{orange}]{path}")
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def get_recurrent_weights(self, *args, **kwargs):
        """
        Get weights of recurrent layer
        """
        raise NotImplementedError(
            "Whem specifying a RNN class you need to define `get_recurrent_weights`"
        )

    def set_recurrent_weights(self, weights):
        """
        Set weights of recurrent layer

        Arguments:
            weights (np.ndarray): (n_units * n_units) 2d array with recurrent weights
        """
        raise NotImplementedError(
            "Whem specifying a RNN class you need to define `set_recurrent_weights`"
        )

    def forward(self, *args):
        """
        Specifies how the forward dynamics of the network are computed
        """
        raise NotImplementedError(
            "Whem specifying a RNN class you need to define `forward`"
        )

    def build(self):
        """
        Update recurrent weights matrix with biological constraints
        """
        iw = npify(self.get_recurrent_weights())
        initializer = RecurrentWeightsInitializer(
            iw,
            dale_ratio=self.dale_ratio,
            autopses=self.autopses,
            connectivity=self.connectivity,
        )
        self.set_recurrent_weights(initializer.weights)

        self._is_built = True

        if self.on_gpu:
            if torch.cuda.is_available():
                print(
                    f"Running on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}"
                )
                self.cuda()
        else:
            print("No GPU found")
            self.on_gpu = False

    def _initialize_hidden(self, x, *args):
        """
        Initialize hidden state of the network
        """
        return torch.zeros((1, x.shape[0], self.n_units))

    def predict_with_history(self, X):
        """
        Predicts a batch of data and keeps
        track of the input and hiddent state at each
        sample along the batch.

        Arguments:
            X (np.ndarray): (batch_size, n_samples, n_inputs) 3d numpy array
                with inputs to use for prediction.

        Returns:
            output_trace (np.ndarray): (n_trials, n_samples, n_outpus) 3d numpy array
                with network's output at each sample
            hidden_trace (np.ndarray): (n_trials, n_samples, n_units) 3d numpy array
                with network's hidden state at each sample
        """
        print(f"[{amber_light}]Predicting input step by step")
        seq_len = X.shape[1]
        n_trials = X.shape[0]

        hidden_trace = np.zeros((n_trials, seq_len, self.n_units))
        output_trace = np.zeros((n_trials, seq_len, self.output_size))

        with base_progress as progress:
            main_id = progress.add_task(
                f"predicting {n_trials} trials",
                start=True,
                total=n_trials,
            )

            # loop over trials in batch
            for trialn in range(n_trials):
                # Progress bar stuff
                trial_id = progress.add_task(
                    f"Trial {trialn}",
                    start=True,
                    total=seq_len,
                )
                progress.update(
                    main_id,
                    completed=trialn,
                )

                # Loop over samples in trial
                h = None
                for step in range(seq_len):
                    progress.update(
                        trial_id,
                        completed=step,
                    )
                    o, h = self(X[trialn, step, :].reshape(1, 1, -1), h)
                    hidden_trace[trialn, step, :] = npify(h)
                    output_trace[trialn, step, :] = npify(o)

                progress.remove_task(trial_id)
            progress.remove_task(main_id)
        return output_trace, hidden_trace

    def predict(self, X):
        """
        Predicts a single trial from a given input.

        Arguments:
            X (np.ndarray): (batch_size, n_samples, n_inputs) 3d numpy array
                with inputs to use for prediction.

        Returns:
            o (np.ndarray): network's output
            h (np.ndarray): network's hidden state
        """
        o, h = self(X[0, :, :].unsqueeze(0))
        o = o.detach().numpy()
        h = h.detach().numpy()
        return o, h
