.PHONY: all test test-aslpv test-dlpv test-lpv graph run run2

test: test-aslpv test-dlpv test-lpv test-kalman

test-aslpv:
	@echo Testing asLPV
	@python3 -m unittest test.lpv.asLPVTest

test-dlpv:
	@echo Testing dLPV
	@python3 -m unittest test.lpv.dLPVTest

test-lpv:
	@echo Testing LPV
	@python3 -m unittest test.lpv.LPVTest

test-kalman:
	@echo Testing HoKalmanIdentifier
	@python3 -m unittest test.lpv.hoKalmanIdentifierTest

graph:
	@echo Graph 1
	@python3 -m src.graphs.graphs

run:
	@echo Executing main
	@python3 -m src.main.main

auto_run: 	
	@echo Execution Auto_main
	@python3 -m src.main.Auto_main
