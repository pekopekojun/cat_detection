https://github.com/balancap/SSD-Tensorflow


# install skicka

~~~
sudo mv /usr/bin/go /usr/bin/go_old
wget https://golang.org/dl/go1.10.1.linux-armv6l.tar.gz
sudo tar -C /usr/local -xzf go1.10.1.linux-armv6l.tar.gz
ls -l /usr/local/go
cat /usr/local/go/VERSION
go get github.com/google/skicka
skicka init
skicka -no-browser-auth df
skicka ls
~~~

# set PATH

~~~
export GOPATH=$HOME/.go
export PATH=$PATH:/usr/local/go/bin;
export PATH=$PATH:/home/pi/.go/bin
~~~


