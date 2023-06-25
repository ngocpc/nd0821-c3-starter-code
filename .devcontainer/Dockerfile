FROM mambaorg/micromamba:1.4.2-jammy AS build

LABEL maintainer="ngoc.phamt@biscglobal.com"

ARG BUILD_ENV

WORKDIR /

# copying dependency definition/setup scripts into container
COPY environment.yaml .

USER root

ENV PATH="/opt/conda/envs/environment/bin:${PATH}"

RUN micromamba create -f environment.yaml -n environment && \
    micromamba install conda-pack -c conda-forge -n environment --freeze-installed && \
    conda-pack --prefix /opt/conda/envs/environment/ --output /tmp/environment.tar.gz && \
    mkdir /venv && \
    tar xf /tmp/environment.tar.gz -C /venv && \
    rm /tmp/environment.tar.gz && \
    /venv/bin/conda-unpack && \
    chown root:root /bin/micromamba

USER micromamba


FROM debian:bullseye-slim AS runtime

USER root
WORKDIR /

ARG BUILD_ENV
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

COPY setup.sh .

RUN apt-get update --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates locales procps --no-install-recommends && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# user environment setup
RUN groupadd -g $USER_GID $USERNAME && \
    useradd -g $USER_GID -l -m $USERNAME -s /bin/bash -u $USER_UID
ENV HOME "/home/$USERNAME"
USER "$USERNAME"
COPY --chown=$USERNAME:$USERNAME --from=build /bin/micromamba /bin/micromamba
COPY --chown=$USERNAME:$USERNAME --from=build /venv /venv
RUN echo "source /venv/bin/activate" >> ~/.bashrc && \
    echo "export PS1='\w:\$ '" >> ~/.bashrc && \
    echo "alias conda='micromamba'" >> ~/.bashrc && \
    echo "git config --global --add safe.directory \$(pwd)" >> ~/.bashrc

# run custom setup, for dependencies not covered by conda/pip
RUN /bin/bash -ci ./setup.sh

WORKDIR "/home/${USERNAME}"
CMD [ "/bin/bash" ]
