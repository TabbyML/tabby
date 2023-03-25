all:

CMD := poetry run python updater.py --data_dir=./testdata/repositories --config_file=testdata/config.toml

test:
	# Cleanup
	rm -rf testdata/repositories/quickjs
	# Initialize
	$(CMD)
	# Updating
	$(CMD)
