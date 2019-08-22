# ScoreSoftware
In this tutorial, we are going to show you how to download and lauch the score sotfware built for our thesis.
## Prerequisities
* If you are using Linux or Mac OS, ignore this point. If you are using Windows 10, you need to install the Ubuntu shell for Windows (Open the Microsoft Store and search: Ubuntu). Once it is installed, open the shell, wait until the installation is finished and create user.
* If you don't have Python3, install it : open the shell and launch `sudo apt-get install python3 python3-pip` (you may need to launch `sudo apt-get update` first)
* You need to install some python libraries. If you didn't installed them yet, do it:
  * Tkinter : `pip3 install tkinter
  * Gensim : `pip3 install gensim
  * Nltk : `pip3 install nltk
  * Enchant : `pip3 install numpy
  * Unidecode : `pip3 install unidecode`
## Download and installation
1. Open a shell and clone this repository : `git clone https://github.com/raphraph1996/ScoreSoftware.git` (you need to use the login and password given in the appendix B).
2. Enter the Score application folder (`cd ScoreApplication`) and launch the install script (`bash install.sh`).
## Launching a model
To launch the software by opening a shell, going to the location of the ScoreApplication folder and typing `python3 gui.py`
