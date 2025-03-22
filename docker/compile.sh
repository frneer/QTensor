#!/bin/bash

# Change to the directory of this script
cd "$(dirname "$0")"

# Make sure build dir exists
mkdir -p ../docs/build

rm -f ../docs/build/*.bbl

# First pass: Generate .aux and .bcf files
docker compose run --rm qtensor_tex \
    pdflatex \
        -interaction=nonstopmode \
        -output-directory=build \
        main.tex && \

# Run biber to generate the bibliography
docker compose run --rm qtensor_tex \
    biber build/plan && \

# Second and third passes to resolve references
docker compose run --rm qtensor_tex \
    pdflatex \
        -interaction=nonstopmode \
        -output-directory=build \
        main.tex && \

docker compose run --rm qtensor_tex \
    pdflatex \
        -interaction=nonstopmode \
        -output-directory=build \
        main.tex

