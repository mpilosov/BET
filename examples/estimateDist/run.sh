#!/bin/bash
python estimatedist.py
python estimatedist_build_pointers.py
python estimatedist_compute_metrics.py
python estimatedist_plotdata.py
