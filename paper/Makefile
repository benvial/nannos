

docker:
	@docker run --rm \
    --volume ${PWD}:/data \
    --user ${(id -u)}:${(id -g)} \
    --env JOURNAL=joss \
    openjournals/inara


clean:
	@rm -f paper.jats
	@rm -f paper.pdf

show:
	xdg-open paper.pdf
