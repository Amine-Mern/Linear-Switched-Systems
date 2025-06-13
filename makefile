.PHONY: all test test-aslpv test-dlpv test-lpv run

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

run:
	@echo Executing main
	@python3 -m src.lpv.main
