![[git原理.png]]

  

add 将文件添加到暂存区（stage）

commit 把暂存区文件提交到本地仓库

checkout 查看指定本地仓库中的文件

push 把本地仓库中的文件推送到远程服务器（如github）中的仓库

pull 从远程仓库获取并合并更改

fetch 会更新本地的远程追踪分支（如 origin/main），但不会自动merge/rebase

clone <url> <optional_dir_name>（将远程仓库的完整副本复制到本地计算机）在本地创建一个新的目录，初始化为 Git 仓库，并将远程仓库的所有文件、分支和提交历史复制到该目录中

init <optional_dir_name> 在当前目录初始化一个本地仓库

mv 移除文件

git config --list 显示当前的Git配置

`git remote show origin`显示远程仓库的信息。

`git remote set-url <remote> <url>`更新远程仓库URL

  

  

  

  

github操作

fork 将他人的仓库复制一份到自己的账户下，形成一个新的远程仓库