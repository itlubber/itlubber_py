## [ITlubber | A Code Monkey](https://itlubber.art)

### python offline deploy
```bash
# 安装依赖
pip install pipreqs
# 在当前目录生成
pipreqs ./ --encoding=utf-8
# 在当前环境下安装依赖
 pip install -r requirements.txt
# 下载当前依赖环境的离线包
pip download -d libs/ --default-timeout=6000 -r requirements.txt -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# 离线安装依赖包 python_lib 为离线包文件的位置
pip install --no-index --find-links=libs -r requirements.txt
```
