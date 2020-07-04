#!/bin/bash
python estimatedist.py
python estimatedist_build_pointers.py
python estimatedist_post-process.py
python estimatedist_plot-post-process.py
