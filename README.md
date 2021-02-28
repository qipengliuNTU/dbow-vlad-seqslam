# vlad-seqslam

1. Reproduce the results in our paper

   ```bash
   $ cd 3_seqslam
   
   $ python plot_PR.py
   ```

2. The complete steps from raw datasets to PR curves

   - Go to the folder `1_vlad`
   - Run the Jupyter Notebook file 
   - Set up all paths according to the instruction inside.
   - Finally generate the confusion matrix mat file
   - Go to the folder `2_dbow`
   - Run `loop-closure` to generate confusion matrix, and then use the script `conver_to_mat.py` to convert the txt file to mat file
   - Go to the folder `3_seqslam`
   - Run the file `loop_closure.py` after setting the paths inside
   - To compare three PR curves, use `plot_PR.py` by setting paths to the pickle files just generated  `loop_closure.py`
   
   

