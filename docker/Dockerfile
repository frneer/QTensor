FROM tensorflow/tensorflow:2.17.0-gpu AS base

# Install python packages
RUN pip3 install \
    matplotlib \
    scikit-learn \
    tensorflow-model-optimization \
    kagglehub

# Set backward compatibility for tfmot
RUN pip3 install tf_keras --no-deps
ENV TF_USE_LEGACY_KERAS=1

# Install linting tools
RUN apt-get update && apt-get install -y \
    git
RUN pip3 install \
    pre-commit==2.20.0

# User setup
ARG USERID
ARG GROUPID
ARG USER
ARG GROUP

RUN apt-get update && apt-get install -y \
    sudo

RUN addgroup --gid $GROUPID $GROUP \
    && adduser --uid $USERID --ingroup $GROUP --home /home/$USER --shell /bin/sh --disabled-password --gecos "" $USER \
    && adduser $USER sudo \
    && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER \
    && chown -R $USER:$GROUP /home/$USER

USER $USER:$GROUP

WORKDIR /home/$USER/qtensor

ENV PYTHONPATH="${PYTHONPATH}:/home/$USER/qtensor/src"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

CMD ["/bin/bash"]
