import matplotlib.pyplot as plt
from msalde.container import ALDEContainer
from msalde.plotter import Plotter

# Dummy config for plot titles
class DummyConfig:
    roc_title = "ROC Curve"
    pr_title = "PR Curve"

def main():
    # Initialize & access repo 
    alde = ALDEContainer("./config/msalde.yaml")
    repo = alde.repository
    
    # Initialize plotter
    plotter = Plotter(DummyConfig(), repo)

    # Replace with the actual simulation or analysis ID you want to plot
    simulation_id = 1

    # Plot ROC and PR curves for the given simulation
    plotter.plot_results(simulation_id, metrics=["roc", "pr"])

if __name__ == "__main__":
    main()
