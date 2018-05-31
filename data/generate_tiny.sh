mkdir train_tiny
find train2017 -maxdepth 1 -type f | head -20000 | xargs cp -t ./train_tiny

mkdir val_tiny
find val2017 -maxdepth 1 -type f |head -2000|xargs cp -t ./val_tiny
