.PHONY: all test test-aslpv test-dlpv test-lpv run

test: test-aslpv test-dlpv test-lpv

test-aslpv:
	python3 -m unittest test.lpv.asLPVTest

test-dlpv:
	python3 -m unittest test.lpv.dLPVTest

test-lpv:
	python3 -m unittest test.lpv.LPVTest

run:
	python3 -m src.lpv.main
