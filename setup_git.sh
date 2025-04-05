#!/bin/bash

# Initialize Git repository
git init

# Add all files to Git
git add .

# Create initial commit
git commit -m "Initial commit: Part Matching Engine"

# Add GitHub remote
git remote add origin https://github.com/pangeafate/partmatchingengine.git

# Push to GitHub
git push -u origin main

echo "Repository has been set up and pushed to GitHub."
echo "Visit https://github.com/pangeafate/partmatchingengine to see your repository."
