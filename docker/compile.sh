#!/bin/bash

mkdir -p build && \
    rm -f build/main.pdf

cd "$(dirname "$0")"

# Compile the main.tex file
docker compose run qtensor_tex \
    pdflatex \
    -interaction=nonstopmode \
    -output-directory=build \
    main.tex
