# Docker file for DSCI_522_Group19_Wine_Quality_Score_Predictor

FROM rocker/tidyverse

RUN apt-get update --fix-missing

# install python3 & virtualenv
RUN apt-get install -y \
    python3-pip \
    && pip3 install virtualenv

# install anaconda & put it in the PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# install r packages
RUN apt-get update -qq && install2.r --error \
    --deps TRUE \
    tidyverse \
    knitr \
    kableExtra \
    docopt


# install python dependencies
RUN conda install docopt \
    requests \
    ipykernel \
    ipython>=7.15 \
    matplotlib>=3.2.2 \
    altair=4.1.* \
    scikit-learn>=1.0 \
    pandas>=1.3.* \
    pandas-profiling>=1.4.3
    
RUN conda install -c conda-forge vega-cli vega-lite-cli

RUN conda install -y pip

RUN pip install rpy2 
RUN pip install docopt==0.6.2 
RUN pip install requests 
RUN pip install imbalanced-learn 
RUN pip install imblearn 
RUN pip install altair_saver
