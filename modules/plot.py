from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import torch

class PlotApp:
    def __init__(self, master, losses, lrs, accs, epochs):
        self.master = master
        self.loss_hist = torch.cat([tens.unsqueeze(0) for tens in losses], dim=0).detach().numpy()
        self.lr_hist = lrs
        self.acc_hist = accs
        self.epochs = epochs

    def plot_acc_loss_epochs(self):
        # Sample data, replace with self.loss_hist and self.lr_hist
        fig = Figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()  # Create a second y-axis

        ax1.plot(range(self.epochs), self.loss_hist, label="Loss", color="cyan")
        ax2.plot(range(self.epochs), self.acc_hist, label="Accuracy", color="red")

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="cyan")
        ax1.set_xlim(min(range(self.epochs)), max(range(self.epochs)))
        ax1.text(self.epochs*.8, 100, f"{(self.acc_hist[-1]*100):.2f}%")

        # Embedding the plot into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Show legends
        ax1.legend(loc="upper left")

    def plot_loss_lr(self):
        # Sample data, replace with self.loss_hist and self.lr_hist
        fig = Figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111)
        #ax2 = ax1.twinx()  # Create a second y-axis

        ax1.plot(self.lr_hist, self.loss_hist, label="Loss", color="cyan")

        ax1.set_xlabel("Learning rate")
        ax1.set_ylabel("Loss", color="cyan")
        ax1.set_xlim(min(self.lr_hist), max(self.lr_hist))

        # Embedding the plot into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Show legends
        ax1.legend(loc="upper left")

    def plot_acc_lr(self):

        # Sample data, replace with self.loss_hist and self.lr_hist
        fig = Figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        #ax2 = ax1.twinx()  # Create a second y-axis

        ax1.plot(self.lr_hist, self.acc_hist, label="Loss", color="cyan")

        ax1.set_xlabel("Learning rate")
        ax1.set_ylabel("Accuracy", color="cyan")
        ax1.set_xlim(min(self.lr_hist), max(self.lr_hist))
        #ax1.set_xticks(self.lr_hist)

        for i, (lr, acc) in enumerate(zip(self.lr_hist, self.acc_hist)):
            if i % 5 == 4:
                ax1.text(lr, acc, f'{lr:.4f}', color='red', fontsize=8)

        # Embedding the plot into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Show legends
        ax1.legend(loc="upper left")


