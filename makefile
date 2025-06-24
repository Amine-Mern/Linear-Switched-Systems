.PHONY: all test test-aslpv test-dlpv test-lpv graph run run2

test: test-aslpv test-dlpv test-lpv

test-aslpv:
	@echo Testing asLPV
	@python3 -m unittest test.lpv.asLPVTest

test-dlpv:
	@echo Testing dLPV
	@python3 -m unittest test.lpv.dLPVTest

test-lpv:
	@echo Testing LPV
	@python3 -m unittest test.lpv.LPVTest

graph:
	@echo Graph 1
	@python3 -m src.graphs.graphs

run:
	@echo Executing main
	@python3 -m src.main.main

run2: 	
	@echo Execution Auto_main
	@python3 -m src.main.Auto_main
