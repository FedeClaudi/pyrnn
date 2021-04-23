import pyinspect as pi
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence  # , pack_padded_sequence
import sys
from myterial import amber_light, orange, salmon
from rich.table import Table

from pyrnn._progress import train_progress, LiveLossPlot
from pyrnn._utils import GracefulInterruptHandler

is_win = sys.platform == "win32"

# ------------------------- Data loader collate func ------------------------ #


def pad_pack_collate(batch):
    (xx, yy) = zip(*batch)

    x_padded = pad_sequence(xx, batch_first=True, padding_value=0)
    y_padded = pad_sequence(yy, batch_first=True, padding_value=0)

    return x_padded, y_padded


# ---------------------------------------------------------------------------- #
#                                    Trainer                                   #
# ---------------------------------------------------------------------------- #


class Trainer:
    def __init__(self):
        return

    def fit(
        self,
        dataset,
        *args,
        batch_size=64,
        n_epochs=100,
        lr=0.001,
        lr_milestones=None,
        gamma=0.1,
        input_length=100,
        l2norm=0,
        stop_loss=None,
        report_path=None,
        num_workers=None,
        plot_live=True,
        logger=None,
        save_at_min_loss=False,
        save_path=None,
        **kwargs,
    ):
        """
        Trains a RNN to predict the data in a dataset.

        Argument:
            dataset (DataSet): instant of torch.utils.data.DataSet subclass
                with training data
            batch_size (int): number of trials per batch
            n_epochs (int): number of training epochs
            lr (float): initial learning rate
            lr_milestones (list): list of epochs numbres at which
                the learning rate should be decreased
            gamma (float): factor by which lr should be reduced at each milestone.
                The updated lr is given by lr * gamma
            input_length (int): number of samples in the input
            l2norm (float): l2 recurrent weights normalization
            stop_loss (float): if not None, when loss <= stop_loss training is stopped
            report_path (str, Path): path to a .txt file where the training report
                will be saved.
            num_workers (int): number of workes used to load the dataset
            plot_live: bool, default True. If true a live plot showing loss is shown
            logger: instance of loguru.logger with filter:  filter=lambda record: "training" in record["extra"]
                see https://github.com/Delgan/loguru for more info
            save_at_min_loss: bool, default False. If true a path must have been passed to save_path.
                If true the RNN is saved every time the lowest loss is achieved.
            save_path: str, default None. If save_at_min_loss is true, use save_path to pass a path to save
                the RNN at.
        """
        stop_loss = stop_loss or -1
        lr_milestones = lr_milestones or [100000000]

        if not self._is_built:
            raise ValueError("Need to first BUILD the RNN model")

        operators = self.setup(
            dataset,
            batch_size,
            lr,
            l2norm,
            lr_milestones,
            gamma,
            num_workers=num_workers,
        )

        losses, epoch_loss, lrates = [], None, []
        with GracefulInterruptHandler() as h:
            with LiveLossPlot(plot_live) as live_plot:
                with train_progress as progress:
                    tid = progress.add_task(
                        "Training",
                        start=True,
                        total=n_epochs,
                        loss=0,
                        lr=lr,
                    )

                    # loop over epochs
                    min_loss = 10000  # start with a very high value
                    for epoch in range(n_epochs + 1):
                        self.on_epoch_start(self, epoch)

                        progress.update(
                            tid,
                            completed=epoch,
                            loss=epoch_loss,
                            lr=lr,
                        )

                        epoch_loss, lr = self.run_epoch(*operators, progress)
                        if epoch_loss is None:
                            break

                        if logger:
                            logger.bind(training=True).info(
                                f"Epoch {epoch} - loss: {epoch_loss:.6f} - lr: {lr:.6f}"
                            )

                        # save if it's the lowest loss so far
                        if save_at_min_loss and epoch_loss < min_loss:
                            if logger:
                                logger.bind(training=True).info(
                                    f"Epoch {epoch} - SAVING MODEL AT LOWEST LOSS. path: {save_path}"
                                )
                            self.save(save_path, overwrite=True)
                            min_loss = epoch_loss

                        losses.append((epoch, epoch_loss))
                        lrates.append(lr)

                        if epoch > 0 and plot_live:
                            live_plot.update([l[1] for l in losses], lrates)

                        if epoch_loss <= stop_loss or h.interrupted:
                            break

        # Make report about training process
        self.report(
            losses,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            lr_milestones=lr_milestones,
            gamma=gamma,
            input_length=input_length,
            l2norm=l2norm,
            stop_loss=stop_loss,
            report_path=report_path,
        )
        return [l[1] for l in losses]

    def setup(
        self,
        dataset,
        batch_size,
        lr,
        l2norm,
        lr_milestones,
        gamma,
        num_workers=None,
    ):
        """
        Sets up stuff needed for training, can be replaced by dedicated methods
        in subclasses e.g. to specify a different optimizer.

        Arguments:
            dataset (DataSet): instant of torch.utils.data.DataSet subclass
                with training data
            batch_size (int): number of trials per batch
            lr (float): initial learning rate
            lr_milestones (list): list of epochs numbres at which
                the learning rate should be decreased
            gamma (float): factor by which lr should be reduced at each milestone.
                The updated lr is given by lr * gamma
            input_length (int): number of samples in the input
            l2norm (float): l2 recurrent weights normalization

        Returns:
            loader: dataset loader (torch.utils.data.DataLoader)
            optimizer: adam optimizer for SGD
            scheduler: learning rate scheduler to decrease lr during training
            criterion: function to compute loss at each epoch.
        """
        if num_workers is None:
            num_workers = 0 if is_win else 2

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=lambda x: np.random.seed(),
            pin_memory=False,
            collate_fn=pad_pack_collate,
        )

        # Get optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=l2norm, amsgrad=True
        )

        # Set up leraning rate scheudler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=gamma
        )

        # Set up training loss
        criterion = torch.nn.MSELoss()

        return loader, optimizer, scheduler, criterion

    def run_epoch(self, loader, optimizer, scheduler, criterion, progress):
        """
        Runs a single training epoch: iterates over
        batches, predicts each batch and computes epoch loss

        Arguments:
            loader: dataset loader
            optimizer: e.g. Adam
            scheduler: lr scheduler
            criterion: function to calculate epoch loss

        Returns:
            epoch_loss (float): loss for this epoch
            lr (float): current learning rate
        """
        # Loop over batch samples
        epoch_loss = 0
        bid = progress.add_task(
            "Batches",
            start=True,
            total=len(loader) - 1,
            loss=None,
            lr=None,
        )

        with GracefulInterruptHandler() as handler:
            for batchn, batch in enumerate(loader):
                progress.update(bid, completed=batchn)

                # initialise
                X, Y = batch
                h = self.on_batch_start(self, X, Y)

                if self.on_gpu:
                    X, Y = X.cuda(0), Y.cuda(0)
                    h = h.cuda(0) if h is not None else h

                # zero gradient
                optimizer.zero_grad()

                # predict
                output, h = self(X, h=h)

                if self.on_gpu:
                    output, h = output.cuda(0), h.cuda(0)

                # backprop + optimizer
                loss = criterion(output, Y)
                loss.backward(retain_graph=True)

                # connectivity constraints
                self.apply_connectivity_constraints()

                # step
                optimizer.step()
                scheduler.step()

                # get current lr
                lr = scheduler.get_last_lr()[-1]

                # update loss
                epoch_loss += loss.item()

                if handler.interrupted:
                    return None, None

        progress.remove_task(bid)
        return epoch_loss, lr

    def apply_connectivity_constraints(self):
        """
        Applies constraints to the connectivity of input,
        recurrent and output layer to ensure that everything's okay
        """
        layers = {
            "input": self.w_in,
            "recurrent": self.w_rec,
            "output": self.w_out,
        }

        for name, constraint in self.connectivity_constraints.items():
            if constraint is not None:
                try:
                    grad = layers[name].weight.grad.clone()
                except AttributeError:
                    # no grad
                    pass
                else:
                    layers[name].weight.grad = grad * np.abs(constraint)

                    if name == "recurrent" and np.any(constraint < 0):
                        # enforce Dale's ratio.
                        excitatory = np.sign(constraint) == 1
                        inhibitory = np.sign(constraint) == -1

                        layers[name].weight.data[excitatory] = torch.abs(
                            layers[name].weight.data[excitatory]
                        )
                        layers[name].weight.data[inhibitory] = -torch.abs(
                            layers[name].weight.data[inhibitory]
                        )

    def report(
        self,
        losses,
        report_path=None,
        **kwargs,
    ):
        """
        Print out a report with training parameters and
        losses history.

        Arguments:
            losses (list): list of (epoch, loss)
            report_path (str, Path): path to .txt file were to save the training report
            kwargs: keyword arguments specify the training parameters
        """
        rep = pi.Report(
            title="Training report",
            color=amber_light,
            accent=salmon,
            dim=orange,
        )
        final = losses[-1]
        rep.add(
            f"[{orange}]Final loss: [b]{round(final[-1], 5)}[/b] after [b]{final[0]}[/b] epochs."
        )

        # Add training params
        tb = Table(box=None)
        tb.add_column(style=f"bold {orange}", justify="right")
        tb.add_column(style=f"{amber_light}", justify="left")

        for param in sorted(kwargs, key=str.lower):
            tb.add_row(param, str(kwargs[param]))
        rep.add(tb, "rich")
        rep.spacer()

        # Save and print
        if report_path is not None:
            srep = pi.utils.stringify(rep, maxlen=-1)
            with open(report_path, "w") as fout:
                fout.write(srep)

        rep.print()
