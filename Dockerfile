FROM continuumio/miniconda
LABEL maintainer="samirak93"

ENV BK_VERSION=1.4.0
ENV PY_VERSION=3.7
ENV NUM_PROCS=4
ENV BOKEH_RESOURCES=cdn

RUN apt-get install git bash
RUN conda config --append channels conda-forge
RUN conda config --append channels districtdatalabs
RUN git clone https://github.com/samirak93/ML_Tool.git /ML

RUN conda install --yes --quiet python=${PY_VERSION} pybase64 pyyaml jinja2 bokeh=${BK_VERSION} numpy numba yellowbrick scipy sympy "nodejs>=8.8" pandas scikit-learn
RUN conda clean -ay

EXPOSE 5006
EXPOSE 80

CMD bokeh serve \
    --allow-websocket-origin="*" \
    --index=/index.html \
    --num-procs=${NUM_PROCS} \
    ML