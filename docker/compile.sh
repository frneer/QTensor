#!/bin/bash

# Create it in docs
cd "$(dirname "$0")"

pushd ../docs
mkdir -p build && \
    rm -f build/main.pdf
popd

# Compile the main.tex file
docker compose run qtensor_tex \
    pdflatex \
    -interaction=nonstopmode \
    -output-directory=build \
    main.tex

