# Fitness tracking reveals task-specific associations between memory, mental health, and exercise

This repository contains data and code used to produce the paper "[Fitness tracking reveals task-specific associations between memory, mental health, and exercise](https://www.biorxiv.org/content/10.1101/2021.10.22.465441v1)" by Jeremy R. Manning, Gina M. Notaro, Esme Chen, and Paxton C. Fitzpatrick.

Additional information on this project can be found [here](https://github.com/ContextLab/brainfit-task). The exact version of the experiment used to acquire our dataset via Amazon Mechanical Turk (sans credentials in `config.txt` and `credentials.json`) may be found [here](https://github.com/ContextLab/brainfit-task/tree/9541c6678ddb1c8da7395ec76869ad97ee1d0dd2).

This repository is organized as follows:
```yaml
root
├── code: all analysis code used in the paper
├── data: all data analyzed in the paper
│   └── task: reference files used to analyze the data
├── docker-setup: configuration files for the Docker image
├── paper: all files needed to generate the main paper and supplement
│   ├── figs: PDF files for all figures in the main paper and supplement
│   │   ├── source: source images used to build the final figures
├── ref: a screen cast of the full experiment
│   └── sandbox_screenrecording_062618.mov
```

Note that compiling the PDFs for the main text and supplement will require you to set up the [Contextual Dynamic Laboratory's Bibliography Management Tool](https://github.com/ContextLab/CDL-bibliography).

## Running our code

To run the included code, used for the analyses presented in our paper, please follow the instructions below.  Note: we've been inconsistent with our internal naming systems for this project; at different points in the project's development and implementation, we have referred to it as FitBrain, BrainFit, and/or FitWit.

### Install dependencies

This package requires Docker (v3) to run. Installation instructions may be found [here](https://docs.docker.com/install/).

1. Clone this repository to your computer
   `git clone https://github.com/ContextLab/brainfit-paper.git`
2. Build an image from the provided `Dockerfile` (from your local repository directory):
   - `docker build -t brainfit-analyses .`
3. Run a container instance from the image, publishing port `8888` and mounting the repository as a volume to access the notebooks. The Jupyter Notebook server will start automatically
  - `docker run --rm -it -p 8888:8888 -v $PWD:/mnt brainfit`
4. Copy the **third** link that appears and paste it into a web browser.

### Running the analyses from the paper

Our main analysis code is organized into the following Jupyter (.ipynb) notebooks:
```yaml
code
├── demographics.ipynb: generate Figure S1 and summarize participant demographics
├── behavioral_data.ipynb: generate Figures 2, 3, S2, S3, S4, and S5
├── fitness_data.ipynb: generate Figures 4, S6, S7, S8, S9, and S10
├── exploratory_analysis_correlations.ipynb: generate Figures 5, S11, S12, S13, and S14, along with the statistical tests associated with the "exploratory correlation analyses" in the main text
└── reverse_correlation_analysis.ipynb: generate Figures 6, S15, S16, S17, S18, S19, and S20
```

Figure 1 (a graphical summary of the experimental tasks) was generated manually, using screen captures from the [experiment](https://github.com/ContextLab/brainfit-task) and/or manually recreated depictions of key events in the experiment.
