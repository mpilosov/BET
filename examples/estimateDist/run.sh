#!/bin/sh
python estimatedist.py
python estimatedist_compute_metrics.py
python estimatedist_plotdata.py
python checkskew_and_plot.py
