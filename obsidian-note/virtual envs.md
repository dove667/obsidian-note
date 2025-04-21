  

conda适合复杂的数据科学计算。pip轻量级，论文常用，对应venv。conda和pip避免一起使用，不过在conda环境里面也可以用pip，但可能会导致依赖配置问题。windows只用pip不用pip3，linux因为自带python2所以一般会指定pip3。复现论文，如果是docker实例其实不用虚拟环境。

流程

打开机器，上传文件（阿里云盘）

连接机器，可以选择ssh连接到本地（建议直接网页jupyter）

创建venv，连接到虚拟环境的python解释器（如果需要虚拟环境）

pip install -r requirements.txt（或者compile requirements.in等）

手动install其他module（如pytorch，不过基础镜像可能自带，不然要跟python解释器、CUDA适配）

下载model和dataset（按照论文方式用国内镜像源下载，或者用奇技淫巧提前下载并修改源码）

python3 xxx 运行实验

  

|   |   |   |   |   |
|---|---|---|---|---|
|**操作**|`**venv**`|`**virtualenv**`|`**pipenv**`|`**poetry**`|
|**创建虚拟环境**|`python -m venv <env_name>`|`virtualenv <env_name>`|`pipenv --python <python_version>`|`poetry new <project_name> && cd <project_name>`|
|**激活虚拟环境**|Windows: `<env_name>\Scripts\activate` Mac/Linux: `source <env_name>/bin/activate`|Windows: `<env_name>\Scripts\activate` Mac/Linux: `source <env_name>/bin/activate`|`pipenv shell`|`poetry shell`|
|**退出虚拟环境**|`deactivate`|`deactivate`|`exit`|`exit`|
|**安装依赖包**|`pip install <package_name>`|`pip install <package_name>`|`pipenv install <package_name>`|`poetry add <package_name>`|
|**卸载依赖包**|`pip uninstall <package_name>`|`pip uninstall <package_name>`|`pipenv uninstall <package_name>`|`poetry remove <package_name>`|
|**列出已安装的包**|`pip list`|`pip list`|`pipenv graph`|`poetry show`|
|**创建依赖文件（requirements.txt）**|`pip freeze > requirements.txt`|`pip freeze > requirements.txt`|`pipenv lock`|`poetry export -f requirements.txt > requirements.txt`|
|**查看虚拟环境位置**|`python -m site`|`virtualenv --version`|`pipenv --venv`|`poetry env info`|
|**删除虚拟环境**|手动删除文件夹|手动删除文件夹|`pipenv --rm`|`poetry env remove`|
|**查看当前虚拟环境的Python版本**|`python --version`|`python --version`|`pipenv --py`|`poetry env info`|

### Conda 常用命令速查表（Cheat Sheet）💡

### 📌 **环境管理**

  

|   |   |
|---|---|
|命令|说明|
|`conda create -n myenv python=3.9`|创建名为 `myenv` 的环境，并安装 Python 3.9|
|`conda env list` 或 `conda info --envs`|查看已创建的环境|
|`conda activate myenv`|激活环境|
|`conda deactivate`|退出当前环境|
|`conda remove -n myenv --all`|删除 `myenv` 环境|

### 📌 **包管理**

|   |   |
|---|---|
|命令|说明|
|`conda install numpy`|安装 `numpy`|
|`conda install numpy pandas matplotlib`|安装多个包|
|`conda install -n myenv numpy`|在指定环境 `myenv` 中安装 `numpy`|
|`conda list`|查看当前环境中的已安装包|
|`conda update numpy`|更新 `numpy`|
|`conda update --all`|更新当前环境中所有包|
|`conda remove numpy`|卸载 `numpy`|
|`conda clean --all`|清理 Conda 缓存，释放空间|

### 📌 **镜像源管理（加速下载）**

|   |   |
|---|---|
|命令|说明|
|`conda config --set show_channel_urls yes`|显示包的下载源|
|`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`|添加清华源|
|`conda config --remove channels https://repo.anaconda.com/pkgs/main/`|移除官方源|
|`conda config --set auto_activate_base false`|禁用 Conda 基础环境的自动激活|

### 📌 **环境克隆与导出**

|   |   |
|---|---|
|命令|说明|
|`conda env export > environment.yml`|导出当前环境配置|
|`conda env create -f environment.yml`|从 `environment.yml` 文件创建环境|
|`conda create --name newenv --clone oldenv`|克隆 `oldenv` 为 `newenv`|

### 📌 **使用** `**pip**` **安装**

|   |   |
|---|---|
|命令|说明|
|`pip install package-name`|在当前 Conda 环境中安装 `package-name`|
|`conda install pip`|确保 Conda 环境中已安装 `pip`|

### 📌 **其他**

|   |   |
|---|---|
|命令|说明|
|`conda info`|查看 Conda 版本及配置信息|
|`conda list --revisions`|查看环境的修改历史|
|`conda install -c conda-forge package-name`|从 `conda-forge` 频道安装包|

如果你有更多 Conda 相关的问题，欢迎随时问我！😆🚀