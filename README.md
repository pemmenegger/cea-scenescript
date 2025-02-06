# SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model
This repository contains inference code for [SceneScript](https://www.projectaria.com/scenescript) with visualisations.

<p align="center"><img src="imgs/scenescript_diagram.png"/></p>

## Using SceneScript on Euler Cluster

SceneScript is set up on the Euler Cluster, allowing you to run computations seamlessly. Since it operates within a shared cluster space, any changes you make will impact all users. To preserve version history and track important results or modifications, consider pushing them to the designated Git repository.

1. Open Jupyter Notebook on Euler Cluster: [https://jupyter.euler.hpc.ethz.ch/hub/spawn](https://jupyter.euler.hpc.ethz.ch/hub/spawn)
2. Navigate to the following directory in the GUI: `/cluster/project/dewolf/pemmenegger/scenescript`
3. Open a terminal in this directory.
4. Load the required Python modules by running: `module load stack/2024-06 python_cuda/3.11.6`
5. Activate the Python virtual environment by running: `source venv/bin/activate`

No you are ready to do computations. Make sure that you always stay at `/cluster/project/dewolf/pemmenegger/scenescript` in your terminal while running jobs.

### Running Jobs on Euler

Do not run .ipynb files via the GUI. Instead, submit jobs via sbatch.
- .ipynb files should be used for debugging and testing.
- Using the `--inplace` flag, the results are written back to the .ipynb file upon success. If an error occurs, the .ipynb file remains unmodified.

Hence, to execute inference.ipynb, use the following command: `sbatch --gpus=1 --mem-per-cpu=8g --wrap="jupyter nbconvert --to notebook --execute inference.ipynb --inplace"`

To execute run_grid_search.py, use: `sbatch --gpus=1 --gres=gpumem:40g --mem-per-cpu=32g --wrap="python run_grid_search.py"`

`--gres=gpumem:40g` refers to the amount of GPU memory you want to use and `--mem-per-cpu=32g`to the amount of CPU memory. You can modify those values as needed.

### Monitoring Jobs

After submitting a job, you can check its status using: `myjobs -j <JOB_ID>`. Job logs will be saved in a Slurm file generated upon submission.

### Closing Jupyter Notebook Session

Once you’re done, properly close your Jupyter Notebook session:

1.	Go to File > Hub Control Panel.
2.	Click Stop Server.
3.	Close the browser window.