#!/usr/bin/env sh

# 确保脚本抛出遇到的错误
set -e

git add -A
git commit -m 'update-dev'
git push -f git@github.com:pxxyyz-dev/SPD-Manifold-Network.git