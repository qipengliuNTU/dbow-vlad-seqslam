# vlad-seqslam

1. Reproduce the results in our paper

   ```bash
   $ cd 2_seqslam
   
   $ python plot_PR.py
   ```

2. The complete steps from raw datasets to PR curves

   - Go to the folder `1_vlad`
   - Run the Jupyter Notebook file 
   - Set up all paths according to the instruction inside.
   - Finally generate the confusion matrix mat file
   - Go to the folder `2_seqslam`
   - Run the file `loop_closure.py` after setting the paths inside
   - To compare two PR curves, use `plot_PR.py` by setting paths to the pickle files generated from two datasets 

   
