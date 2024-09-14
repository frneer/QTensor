#!/bin/sh

# Remove unused imports with autoflake
autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive .

# Sort imports with isort
isort .

# Format docstrings with docformatter
find . -name "*.py" -exec docformatter --in-place --wrap-summaries 79 --wrap-descriptions 79 {} +

# Format code with Black
black .
