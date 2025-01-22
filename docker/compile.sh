#!/bin/bash

# Change to the directory of this script
cd "$(dirname "$0")"

# Make sure build dir exists
mkdir -p ../docs/build

# Compile the main.tex file
docker compose run qtensor_tex \
    pdflatex \
    -interaction=nonstopmode \
    -output-directory=build \
    main.tex
