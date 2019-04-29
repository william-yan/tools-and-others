1. ubuntu 下安装 typora 

   （1） or run:

   //sudo apt-key adv --keyserver keyserver.ubuntu.com--recv-keys BA300B7755AFCFAE

   wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -

   

   # add Typora's repository
   sudo add-apt-repository 'deb https://typora.io/linux ./'
   sudo apt-get update

   # install typora
   sudo apt-get install typora

2. 说明：尝试过其他方法，但是 一直出现文件 list 第 52 行的记录格式有误 /etc/apt/sources.list (Component)，上面的这个方法是可以完美安装的