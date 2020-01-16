# A detailed installation process of DeepKnee
Tested on a fresh Ubuntu 18.04 LTS

## Installing Docker

1. Install the pre-requisites for `apt` and get the repository added to your list of repositories.
```
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
```
Please note that if you use Linux Mint (also Ubuntu based), you might need to add the repo this way.
You need to replace the last two lines in the above script.

```
echo -e "\ndeb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" | sudo tee -a /etc/apt/sources.list
sudo apt update
```

2. Installing docker and checking that it works
```
sudo apt install -y docker-ce
sudo docker run hello-world
```

The lines above will install the docker itself and will also run the test `hellow-world` container.

3. Making sure that docker can be executed w/o root:
```
sudo usermod -aG docker ${USER}
newgrp docker 
```

4. Install docker-compose. The easiest path is to use anaconda:
```
cd
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x miniconda.sh
./miniconda.sh
```

Note: if you use zsh instead of bash, you need to modify your `.zshrc` in order to proceed by adding the following at the end of the file:

```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/lext/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/lext/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lext/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/lext/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

Exit your terminal and open it again. Now you should have a `base` conda environment activated.
Install docker-compose:

```
pip install docker-compose
```
5. Install DeepKnee (has to be run in the root of this repo):
```
sh deploy.sh cpu
```