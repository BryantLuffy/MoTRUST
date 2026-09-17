# 上传到 GitHub

解压后，以同时包含 `README.md`、`pyproject.toml` 和 `src/` 的目录作为仓库根目录。
不要在 GitHub 根目录再套一层项目文件夹，也不要只上传 ZIP。

在 GitHub 创建空仓库，然后使用 GitHub Desktop 发布这个目录；也可以在目录内执行：

```bash
git init
git add .
git commit -m "Add MoTRUST"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MoTRUST.git
git push -u origin main
```

将 `YOUR_USERNAME` 替换为自己的账号。保留 `.gitignore` 和 `.github/` 隐藏文件。

此目录是独立的 MoTRUST 软件项目。数据读取、评价和各工作流已归入
`src/motrust/`，数据、模型权重与运行结果另行管理。安装与运行说明见首页和
`docs/workflows.md`。本次整理只生成本地文件，没有创建或推送远程仓库。
