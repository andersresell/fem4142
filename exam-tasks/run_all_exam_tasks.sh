#!/bin/bash
# Script to run all exam task scripts and generate figures

echo "============================================================"
echo "Running all exam task scripts"
echo "============================================================"

# Navigate to the project directory
cd /home/anders/projects/fem4146

echo ""
echo "------------------------------------------------------------"
echo "1/3: Running beam comparison..."
echo "------------------------------------------------------------"
MPLBACKEND=Agg python exam-tasks/beam_comparison.py

echo ""
echo "------------------------------------------------------------"
echo "2/3: Running convergence study (tip load)..."
echo "------------------------------------------------------------"
MPLBACKEND=Agg python exam-tasks/convergence_tipload_FINAL.py

echo ""
echo "------------------------------------------------------------"
echo "3/3: Running plate locking study..."
echo "------------------------------------------------------------"
MPLBACKEND=Agg python exam-tasks/plate_locking.py

echo ""
echo "============================================================"
echo "All exam task scripts completed!"
echo "============================================================"
echo ""
echo "Generated figures:"
echo "  - exam-tasks/beam_comparison.png"
echo "  - exam-tasks/exam_plot_FINAL_UNLABELED.png"
echo "  - exam-tasks/exam_plot_FINAL_ANSWER.png"
echo "  - exam-tasks/plate_locking_comparison.png"
echo "============================================================"
