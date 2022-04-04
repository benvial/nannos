#!/usr/bin/env sh

# Originally from https://github.com/latex3/latex3

# This script is used for building LaTeX files using Travis
# A minimal current TL is installed adding only the packages that are
# required

export PATH=/root/texlive/bin/x86_64-linux:$PATH

# Obtain TeX Live
wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz
cd install-tl-20*

# Install a minimal system
./install-tl --profile ../texlive.profile

cd ..

tlmgr install xetex

# We specify the directory for texlive_packages
tlmgr install $(sed 's/\s*#.*//;/^\s*$/d' texlive_packages)

# Keep no backups (not required, simply makes cache bigger)
tlmgr option -- autobackup 0

# Update the TL install but add nothing new
tlmgr update --self --all --no-auto-install
