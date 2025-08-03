# install uv package With pip
pip install uv

# check install
uv
# it should show you information about uv and its options

# create project
uv init 
# it creates project files for it. name will be the same as the directory name. we can add a name with uv init <name>

# create venv 
uv venv
source .venv/bin/activate

# add packages
uv add langchain

# add packages from requirements.txt
uv add -r requirements.txt