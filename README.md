# dc-assignments


## installation
0. you can avoid the next steps if you install the dependencies with pip (not recommended): `pip install -r requirements.txt`
1. install poetry with your package manager or with `pip install -U poetry`
2. setup python environment with poetry: `poetry install --sync`

## run
main files are in the `main` folder because `src` only contains classes and methods.

test files are in the `test` folder. You can use `--multiprocessing` to split them across multiple processes. Results will be saved in the `results` folder

0. you can skip the next steps if you use `poetry run python3 {target_script}.py` or if you installed the dependencies directly with pip you can go to step 2
1. activate the python environment: `poetry shell`
2. run: `python3 {target_script}.py`

### storage
config files are in the `config` folders.

`--extension` flag selects what class it should use between `base` (base extension) and `advanced`. If no extension is provided it will run the stock one.

### examples
`poetry run python3 main/mmn.py`

`poetry run python3 test/mmn.py --multiprocessing`

`poetry run python3 main/storage.py config/p2p.cfg --max-t "2 years"`

`poetry run python3 test/storage.py config/p2p.cfg --multiprocessing`

### docker
for better testing a docker image has been defined and you can use it with the `docker-compose.yml` with: `docker-compose up -d {mmn-test|storage-test}`