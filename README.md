# ProLaTherm: *Pro*tein *La*nguage Model-based *Therm*ophilicity Predictor
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-388/)

In this repository, we publish three parts related to protein thermophilicity prediction:

- Code to get predictions using ProLaTherm, the first protein language model-based thermophilicity predictor
- A new benchmark dataset for protein thermophilicity prediction consisting of three significantly updated datasets from literature and newly collected data
- Code to run our protein thermophilicity prediction optimization framework including several feature- and sequence-based prediction models

## Predict using ProLaTherm
If you want to get thermophilicity predictions for your protein sequences using ProLaTherm, you first have to prepare a .fasta file containing sample ids and amino acid sequences.
We expect the standard fasta format with the sample id, and the amino acid sequence following in the next line, see `prolatherm\assets\dummy_data.fasta` for an example.

Then you have two possibilities to run ProLaTherm:
- Create a Docker container using the Dockerfile we provide and run ProLaTherm within this tested working  environment
- Run ProLaTherm directly on your machine (e.g. within a virtual environment to prevent Python package version changes) after installing all required packages using our requirements.txt

Both possibilities are outlined below. In either case, make sure to have a minimum of 20GB memory. As you only run inference, a GPU is not necessary, but if you want to use one it should have at least 16GB memory.
Using the provided .fasta-file, the pipeline will run on the command line and create a .csv-file containing the results.

### Docker workflow
Docker needs to be installed and running on your machine, see the [Installation Guidelines at the Docker website](https://docs.docker.com/get-docker/)
On Ubuntu, you can use ``docker run hello-world`` or ``docker --version`` to check if Docker works
(Caution: add sudo if you are not in the docker group).

If you want to use GPU support, you need to install [nvidia-docker-2](https://github.com/NVIDIA/nvidia-docker), see this [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
and a version of CUDA >= 11.2 (see this [CUDA Installation Guide](https://docs.nvidia.com/cuda/index.html#installation-guides). To check your CUDA version, just run `nvidia-smi` in a terminal.

1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository

        git clone https://github.com/grimmlab/ProLaTherm.git

3. Navigate to `prolatherm/Docker` after cloning the repository

        cd ProLaTherm/prolatherm/Docker

4. Build a Docker image using the provided Dockerfile tagged with the IMAGENAME of your choice

        docker build -t IMAGENAME .

5. Run an interactive Docker container based on the created image with a CONTAINERNAME of your choice

        docker run -it -v /PATH/TO/REPO/FOLDER:/REPO_DIRECTORY/IN/CONTAINER -v /PATH/TO/DATA/DIRECTORY:/DATA_DIRECTORY/IN/CONTAINER -v /PATH/TO/RESULTS/SAVE/DIRECTORY:/SAVE_DIRECTORY/IN/CONTAINER --name CONTAINERNAME IMAGENAME

    - Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directory where you want to save your results using the option ``-v``.
    - You can restrict the number of cpus using the option ``cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP``.
    - Specify a gpu device using ``--gpus device=DEVICE_NUMBER`` if you want to use GPU support.

    Let's have a look at an example. We assume hat you created a Docker image called ``prolatherm_image``, your repository and data is placed in (subfolders of) ``/myhome/``, you want to save your results to ``/myhome/`` (so ``/myhome/`` is the only directory you need to mount in your container), you only want to use CPUs 0 to 9 and GPU 0 and you want to call your container ``prolatherm_cont``. Then you have to run the following command:

        docker run -it -v /myhome/:/myhome_in_my_container/ --cpuset-cpus 0-9 --gpus device=0 --name prolatherm_cont prolatherm_image

6. Navigate to the directory where the repository is placed within your container and to the `prolatherm` subfolder

        cd /REPO_DIRECTORY/IN/CONTAINER/ProLaTherm/prolatherm

7. Run ProLaTherm with giving the full path to the .fasta-file (default: `prolatherm/assets/dummy_fasta.csv`) and directory where you want to save your results file (default: repository folder)

        python3 run_prolatherm.py -df /FULL/PATH/TO/FASTA/FILE -sd /FULL/PATH/TO/RESULTS/SAVE/DIR
In case you have problems in getting inference running on your GPU, we provide an option `--no_gpu`, which you can set to `True` when calling `python3 run_prolatherm.py`

That's it! The .fasta-file will be processed in batches of 10 samples, you will see the current status on the command line and in the end a .csv-file containing the results will be created.

### Run ProLaTherm directly on your machine
Instead of using Docker, you can run ProLatherm directly on your machine. We recommend to set up a Python virtual environment for that purpose, see here: https://docs.python.org/3/library/venv.html
ProLaTherm was developed and tested in Python 3.8, so please use this Python version

For our tutorial below, we assume that you know how to work with a virtual environment and are working within such an environment (or you decided against it).

1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository

        git clone https://github.com/grimmlab/ProLaTherm.git

3. Navigate to `prolatherm/Docker` after cloning the repository

        cd ProLaTherm/prolatherm/Docker

4. Install all required Python packages using our `requirements.txt`

        pip3 install -r requirements.txt

5. Navigate to the `prolatherm` subfolder of your repository

        cd ..

6. Run ProLaTherm with giving the full path to the .fasta-file (default: `prolatherm/assets/dummy_fasta.csv`) and directory where you want to save your results file (default: repository folder)

        python3 run_prolatherm.py -df /FULL/PATH/TO/FASTA/FILE -sd /FULL/PATH/TO/RESULTS/SAVE/DIR
In case you have problems in getting inference running on your GPU, we provide an option `--no_gpu`, which you can set to `True` when calling `python3 run_prolatherm.py`

That's it! The .fasta-file will be processed in batches of 10 samples, you will see the current status on the command line and in the end a .csv-file containing the results will be created.


## Data
Here, we publish our new benchmark dataset for protein thermophilicity prediction consisting of significantly updated datasets from literature and newly collected data. 
For details on the data collection and preprocessing, see our publication below.

Within the folder `data`, one can find three subfolders:

- `datasets_w_datasplits`: preprocessed and feature-engineered datasets used for our scientific paper (see below) as .csv-files and so-called *index_files* that contain indices for our datasplits to ensure reproducibility. Both files are generated by our optimization pipeline, when providing fasta files containing thermophilic and non-thermophilic proteins
- `fasta_files`: .fasta-files containing our benchmark data split in files containing *thermo* and *non_thermo* species, both as *_fulldata* and *_preprocessed* (CD-Hit with threshold of 40%, removing sequences with lengths below and above 5th and 95th percentile, see our publication for details)
  - Files with prefix *ProtThermoPred_shared_organisms_*: protein sequences of species that occur within the three previously published datasets and our newly collected data
  - Files with prefix *ProtThermoPred_unique_organisms_*: protein sequences of species that only occur in our newly collected data
  - Further .fasta-files with prefix *ProtThermoPred_* (not containing key words *shared_organisms* and *unique_organisms*): protein sequences of our full dataset without a species-specific split
  - Subfolder `update_of_benchmark_datasets`: .fasta-files containing significantly updated datasets from literature
- `meta_files`: .csv-files with meta information such as species and UniProt history for all datasets


## Protein Thermophilicity Prediction Optimization Framework
To run our optimization framework, we highly recommend our workflow using Docker due to its easy-to-use interface and ready-to-use working environment
within a Docker container.

This optimization framework is based on easyPheno, our phenotype prediction optimization framework. 
So in case you are interested in further details, easyPheno's comprehensive documentation including installation guides, tutorials and much more, might be helpful: https://easypheno.readthedocs.io/


### Requirements for the optimization framework
Docker needs to be installed and running on your machine, see the [Installation Guidelines at the Docker website](https://docs.docker.com/get-docker/)
On Ubuntu, you can use ``docker run hello-world`` or ``docker --version`` to check if Docker works
(Caution: add sudo if you are not in the docker group).

If you want to use GPU support, you need to install [nvidia-docker-2](https://github.com/NVIDIA/nvidia-docker), see this [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
and a version of CUDA >= 11.2 (see this [CUDA Installation Guide](https://docs.nvidia.com/cuda/index.html#installation-guides). To check your CUDA version, just run `nvidia-smi` in a terminal.

CAUTION: The models *transformer* and *bigbird* require a minimum GPU memory of 48GB. 

### Setup of the optimization framework
1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository

        git clone https://github.com/grimmlab/ProLaTherm.git

3. Navigate to `thermpred/Docker` after cloning the repository

        cd ProLaTherm/thermpred/Docker

4. Build a Docker image using the provided Dockerfile tagged with the IMAGENAME of your choice

        docker build -t IMAGENAME .

5. Run an interactive Docker container based on the created image with a CONTAINERNAME of your choice

        docker run -it -v /PATH/TO/REPO/FOLDER:/REPO_DIRECTORY/IN/CONTAINER -v /PATH/TO/DATA/DIRECTORY:/DATA_DIRECTORY/IN/CONTAINER -v /PATH/TO/RESULTS/SAVE/DIRECTORY:/SAVE_DIRECTORY/IN/CONTAINER --name CONTAINERNAME IMAGENAME

    - Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directory where you want to save your results using the option ``-v``.
    - You can restrict the number of cpus using the option ``cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP``.
    - Specify a gpu device using ``--gpus device=DEVICE_NUMBER`` if you want to use GPU support.

    Let's have a look at an example. We assume hat you created a Docker image called ``thermpred_image``, your repository and data is placed in (subfolders of) ``/myhome/``, you want to save your results to ``/myhome/`` (so ``/myhome/`` is the only directory you need to mount in your container), you only want to use CPUs 0 to 9 and GPU 0 and you want to call your container ``thermpred_cont``. Then you have to run the following command:

        docker run -it -v /myhome/:/myhome_in_my_container/ --cpuset-cpus 0-9 --gpus device=0 --name thermpred_cont thermpred_image 

Your setup is finished!

### Run the optimization framework with prepared data
You are at the **root directory within your Docker container**, i.e. after step 5 of the above-described setup.

If you closed the Docker container that you created at the end of the installation, just use ``docker start -i CONTAINERNAME``
to start it in interactive mode again. If you did not create a container yet, go back to above-described setup.

1. Navigate to the directory where the repository is placed within your container

        cd /REPO_DIRECTORY/IN/CONTAINER/ProLaTherm

2. Run thermpred (as module). By default, thermpred starts the optimization procedure for 10 trials with XGBoost and a 5-fold nested cross-validation using the `ProtThermPred_fulldataset.csv` we provide in `data/datasets_w_datasplits`.

        python3 -m thermpred.run --save_dir SAVE_DIRECTORY

    That's it! You can now find the results in the save directory you specified. By default, if you do not specify a `save_dir` a results folder will be created at the top directory of your repository.

3. To get an overview of the different options you can set for running thermpred, just do:

        python3 -m thermpred.run --help


Feel free to test thermpred, e.g. with other prediction models.

CAUTION: If you want to run the optimization for `prolatherm`, you first have to generate .h5-files containing the embeddings. This will need up to 20GB of memory on your machine!
To generate such a file, you need to run `python3 -m thermpred.generate_pretrained_embeddings` from the directory where your repository is placed in the Docker container.
To check the different options, e.g. to change the dataset for which we generate the embeddings by default, please run `python3 -m thermpred.generate_pretrained_embeddings --help`.


### Run the optimization framework using your own .fasta-files
To run the optimization framework with your own data, you have to prepare two .fasta-files: one containing the thermophilic and one containing the non-thermophilic proteins.
We expect the standard .fasta-format with the identifier of the sample, followed by a line break and the amino acid sequence given in the single letter code in the next line.

You are at the **root directory within your Docker container**, i.e. after step 5 of the above-described setup.

If you closed the Docker container that you created at the end of the installation, just use ``docker start -i CONTAINERNAME``
to start it in interactive mode again. If you did not create a container yet, go back to above-described setup.

1. Navigate to the directory where the repository is placed within your container

        cd /REPO_DIRECTORY/IN/CONTAINER/ProLaTherm

2. Run thermpred (as module). By default, thermpred starts the optimization procedure for 10 trials with XGBoost and a 5-fold nested cross-validation using your newly created dataset.

        python3 -m thermpred.run -dd /DIRECTORY/CONTAINING/YOUR/FASTAFILES -sd /SAVEDIRECTORY/FOR/CREATED/DATASET -fth FULL_NAME_OF_THERMOPHILIC_FASTA_FILE -fnth FULL_NAME_OF_NON_THERMOPHILIC_FASTA_FILE -nnd NAME_OF_THE_NEW_DATASET

You only have to invoke the data preprocessing once. For the next run, you can use the above-described workflow for prepared data, and need to specify the name of your new dataset as the dataset to use.

#### Useful Docker commands

The subsequent Docker commands might be useful when using thermpred.
See [here](https://docs.docker.com/engine/reference/commandline/docker/) for a full guide on the Docker commands.

- `docker images`: List all Docker images on your machine
- `docker ps`: List all running Docker containers on your machine
- `docker ps -a`: List all Docker containers (including stopped ones) on your machine
- `docker start -i CONTAINERNAME`: Start a (stopped) Docker container interactively to enter its command line interface

## Contributors
This pipeline is developed and maintained by members of the [Bioinformatics lab](https://bit.cs.tum.de) lead by [Prof. Dr. Dominik Grimm](https://bit.cs.tum.de/team/dominik-grimm/):
- [Florian Haselbeck, M.Sc.](https://bit.cs.tum.de/team/florian-haselbeck/)
- [Maura John, M.Sc.](https://bit.cs.tum.de/team/maura-john/)
- [Jonathan Pirnay, M.Sc.](https://bit.cs.tum.de/team/jonathan-pirnay/)

## Citation
When using parts of this repository, please cite our publication:

**Superior Protein Thermophilicity Prediction With Protein Language Model Embeddings**  
F Haselbeck, M John, Y Zhang, J Pirnay, JP Fuenzalida-Werner, RD Costa, and DG Grimm  
*currently under review*

Keywords: Protein Thermophilicity Prediction, Protein Language Model, Machine Learning
