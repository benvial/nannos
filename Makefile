
SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: clean lint req doc help dev

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = nannos
PYTHON_INTERPRETER = python3
HOSTING = gitlab
VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")
BRANCH=$(shell git branch --show-current)
URL=$(shell python3 -c "import nannos; print(nannos.__website__)")
LESSC=$(PROJECT_DIR)/doc/node_modules/less/bin/lessc
GITLAB_PROJECT_ID=28703132
GITLAB_GROUP_ID=12956132
LINT_FLAGS=E501,F401,F403,F405,W503,E402,E203

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifdef TEST_PARALLEL
TEST_ARGS=-n auto #--dist loadscope
endif


message = @make -s printmessage RULE=${1}

printmessage:
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/^/---/" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} | grep "\---${RULE}---" \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=0 \
		-v col_on="$$(tput setaf 4)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s ", col_on, -indent, ">>>"; \
		n = split($$3, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i] ; \
		} \
		printf "%s ", col_off; \
		printf "\n"; \
	}'

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
req:
	$(call message,${@})
	pip install -r requirements.txt

## Install Python dependencies for dev and test
dev:
	@pip install -r dev/requirements.txt

## Clean generated files
cleangen:
	$(call message,${@})
	@find . -not -path "./test/data/*" | grep -E "(__pycache__|\.pyc|\.ipynb_checkpoints|\.pyo$\)" | xargs rm -rf
	@rm -rf .pytest_cache  build/ dist/ tmp/ htmlcov/ #nannos/nannos.egg-info/

## Clean documentation
cleandoc:
	$(call message,${@})
	@cd doc && make -s clean

## Clean project
clean: cleantest cleangen cleanreport cleandoc
	$(call message,${@})

## Lint using flake8 (sources only)
lint:
	$(call message,${@})
	@flake8 --exit-zero --ignore=$(LINT_FLAGS) nannos/$(PROJECT_NAME)
	
## Lint using flake8
lint-all:
	$(call message,${@})
	@flake8 --exit-zero --ignore=$(LINT_FLAGS) nannos/$(PROJECT_NAME) test/ examples/ --exclude "dev*"

## Check for duplicated code
dup:
	$(call message,${@})
	@pylint --exit-zero -f colorized --disable=all --enable=similarities nannos/$(PROJECT_NAME)


## Clean code stats
cleanreport:
	$(call message,${@})
	@rm -f pylint.html

## Report code stats
report: cleanreport
	$(call message,${@})
	@pylint nannos/$(PROJECT_NAME) | pylint-json2html -f jsonextended -o pylint.html


## Check for missing docstring
dcstr:
	$(call message,${@})
	@pydocstyle nannos/$(PROJECT_NAME)  || true

## Metric for complexity
rad:
	$(call message,${@})
	@radon cc nannos/$(PROJECT_NAME) -a -nb

## Run all code checks
code-check: lint dup dcstr rad
	$(call message,${@})

## Reformat code
style:
	$(call message,${@})
	@isort -l 88 .
	@black -l 88 .


## Push to gitlab
gl:
	$(call message,${@})
	@git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG"
	@git push origin $(BRANCH)


## Show gitlab repository
repo:
	$(call message,${@})
	xdg-open https://gitlab.com/nannos/nannos


## Clean, reformat and push to gitlab
save: style gl
	$(call message,${@})


## Push to gitlab (skipping continuous integration)
gl-noci:
	$(call message,${@})
	@git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG [skip ci]"
	@git push origin $(BRANCH)



## Clean, reformat and push to gitlab (skipping continuous integration)
save-noci: style gl-noci
	$(call message,${@})



## Make documentation css
less:
	$(call message,${@})
	@rm -f doc/_custom/static/css/*.css
	@cd doc/_custom/static/css/less && \
	chmod +x make_css && ./make_css $(LESSC)

## Rebuild css on change
watch-less:
	$(call message,${@})
	while inotifywait -e close_write ./doc/_custom/static/css/less/*.less; do make -s less; done


## Install requirements for building documentation
doc-req:
	$(call message,${@})
	@cd doc && pip install --upgrade -r requirements.txt && npm install lessc


## Build html documentation (only updated examples)
doc: less
	$(call message,${@})
	@cd doc && PYVISTA_OFF_SCREEN=true make -s html && make -s postpro-html


## Build html documentation (without examples)
doc-noplot: less
	$(call message,${@})
	@cd doc && make -s clean && make -s html-noplot && make -s postpro-html

## Show locally built html documentation in a browser
show-doc:
	$(call message,${@})
	@cd doc && make -s show

## Show locally built pdf documentation
show-pdf:
	$(call message,${@})
	@cd doc && make -s show-pdf

## Build pdf documentation (only updated examples)
pdf:
	$(call message,${@})
	@cd doc && make -s latexpdf

## Build pdf documentation (without examples)
pdf-noplot:
	$(call message,${@})
	@cd doc && make -s latexpdf-noplot

## Clean test coverage reports
cleantest:
	$(call message,${@})
	@rm -rf .coverage* htmlcov coverage.xml


## Install requirements for testing
test-req:
	$(call message,${@})
	@cd test && pip install --upgrade -r requirements.txt



define runtest
		@echo
		@echo "-----------------------------------------------------------------------------"
		@echo "-----------------------    testing $(1) backend    --------------------------"
		@echo "-----------------------------------------------------------------------------"
		@echo
		@export MPLBACKEND=agg && export NANNOS_BACKEND=$(1) && \
		pytest ./test/basic \
		--cov=nannos/$(PROJECT_NAME) --cov-append --cov-report term \
		--durations=0 $(TEST_ARGS)
endef

## Run the test suite with numpy backend only
test-numpy:
	$(call message,${@})
	$(call	runtest,numpy)


## Run the test suite with scipy backend only
test-scipy:
	$(call message,${@})
	$(call	runtest,scipy)


## Run the test suite with autograd backend only
test-autograd:
	$(call message,${@})
	$(call	runtest,autograd)


## Run the test suite with jax backend only
test-jax:
	$(call message,${@})
	$(call	runtest,jax)


## Run the test suite with torch backend only
test-torch:
	$(call message,${@})
	$(call	runtest,torch)

## Run tests with different backends
test-allbk: test-numpy test-scipy test-autograd test-jax test-torch
	$(call message,${@})

## Run the commmon tests
test-common:
	$(call message,${@})
	@export MPLBACKEND=agg && export NANNOS_BACKEND=numpy && pytest ./test/common \
	--cov=nannos/$(PROJECT_NAME) --cov-append --cov-report term \
	--cov-report html --cov-report xml --durations=0 $(TEST_ARGS)

## Run the test suite
test: cleantest test-allbk test-common
	$(call message,${@})


## Run the test suite (parallel)
testpara: cleantest
	$(call message,${@})
	@export OMP_NUM_THREADS=1 && make -s test TEST_PARALLEL=1

## Copy the coverage html into documentation
covdoc:
	$(call message,${@})
	@ls doc/_build/html/ || make doc
	@ls htmlcov/ || make -s test && mv htmlcov/ doc/_build/html/coverage/

## Install locally
install:
	$(call message,${@})
	pip install -e .


## Tag and push tags
tag: clean style
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Version v$(VERSION)"
	@git add -A
	git commit -a -m "Publish v$(VERSION)"
	@git push origin $(BRANCH)
	@git tag v$(VERSION) || echo Ignoring tag since it already exists
	@git push --tags || echo Ignoring tag since it already exists on the remote

## Create a release
release: tag
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@gitlab project-release create --project-id $(GITLAB_PROJECT_ID) \
	--name "version $(VERSION)" --tag-name "v$(VERSION)" --description "Released version $(VERSION)"


## Create python package
package:
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@rm -f dist/*
	@python3 -m build --sdist --wheel .

## Upload to pypi
pypi: package
	$(call message,${@})
	@twine upload dist/*

## Make checksum for release
checksum:
	$(call message,${@})
	@echo v$(VERSION)
	$(eval SHA256=$(shell curl -sL https://gitlab.com/nannos/nannos/-/archive/v$(VERSION)/nannos-v$(VERSION).tar.gz | openssl sha256 | cut -c17-))
	@echo $(SHA256)


## Make checksum for release
checksum-ci:
	$(call message,${@})
	@echo v$(VERSION)
	$(eval SHA256=$(shell curl -sL https://gitlab.com/nannos/nannos/-/archive/v$(VERSION)/nannos-v$(VERSION).tar.gz | openssl sha256 | cut -c16-))
	@echo $(SHA256)

## Update conda-forge recipe
recipe: 
	$(call message,${@})
	cd .. && rm -rf nannos-feedstock && \
	git clone git@github.com:benvial/nannos-feedstock.git && cd nannos-feedstock  && \
	sed -i "s/sha256: .*/sha256: $(SHA256)/" recipe/meta.yaml && \
	sed -i "s/number: .*/number: 0/" recipe/meta.yaml && \
	sed -i "s/{% set version = .*/{% set version = \"$(VERSION)\" %}/" recipe/meta.yaml


# git branch v$(VERSION) && git checkout v$(VERSION) && \


## Update conda-forge package
conda: checksum recipe
	$(call message,${@})
	cd ../nannos-feedstock && \
	git add . && \
	git commit -a -m "New version $(VERSION)" && \
	git push origin v$(VERSION) --force && echo done

## Update conda-forge package
conda-ci: checksum-ci recipe
	$(call message,${@})
	cd ../nannos-feedstock && \
	git add . && \
	git commit -a -m "New version $(VERSION)" && \
	git push origin v$(VERSION) --force && echo done


## Publish release on pypi and conda-forge
publish: tag release pypi conda
	$(call message,${@})

## Update header text
header:
	$(call message,${@})
	@cd dev && python update_header.py


## Download and install fonts locally
fonts:
	$(call message,${@})
	@rm -rf ~/.cache/matplotlib && \
	bash doc/binder/postBuild

## Copy matplotlib stylesheet
stylesheet:
	$(call message,${@})
	@mkdir -p ${HOME}/.config/matplotlib && \
	cp doc/_custom/nannos.mplstyle matplotlibrc ${HOME}/.config/matplotlib/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################


# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

help:
	@echo -e "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo -e
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
