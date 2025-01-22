FROM texlive/texlive:latest

# Install additional packages
RUN tlmgr install \
    latexmk \
    xcolor \
    geometry \
    amsmath \
    hyperref \
    fontspec

WORKDIR /workspace

CMD ["/bin/bash"]
