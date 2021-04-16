---
layout: post
comments: true
title: "How to shoot a time elapse with Raspberry pi"
excerpt: "I'll note down the steps to set up a raspberry pi for time-elapse
shots."
date:   2021-04-16 01:00:00
mathjax: true
---

In the contemporary life, speed is becoming a primary driving force behind productivity, which indeed helps people to achieve their goals and dreams. However we start to forget about the simple beauty around our daily life, like the sunshine, the fresh air, etc.

Ever since I got interested in planting (mainly vegetables), I am touching and feeling the power of nature (soil). After came cross a few Youtube videos, where the plant growing process is time-elapsed, I decide to explore this fascinating world by myself. Therefore I set up my Raspberry pi to do my own experiments. Hopefully interesting results will delivered in the coming months.


### Install Raspberry Pi OS
> https://www.raspberrypi.org/software


### Camera module
- Install the camera module on the raspberry pi


<div class="imgcap">
<img src="/assets/time_elapse/install_rasp.png" height="300">
<div class="thecap">Before and after assembly.</div>
</div>


- Enable the camera on the raspberry pi

```bash
sudo raspi-config
# -> Interfacing Option -> Camera
reboot
```

- Testing the camera

```bash
raspistill -v -o test.jpg
```

### WIFI
- Install WIFI driver for TP-LINK TL-WN725N V2

> https://www.raspberrypi.org/forums/viewtopic.php?f=28&t=62371

```bash
sudo wget http://downloads.fars-robotics.net/wifi-drivers/install-wifi -O /usr/bin/install-wifi
sudo chmod +x /usr/bin/install-wif
reboot
```

- Add your WIFI information

> https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md

```bash
sudo vi /etc/wpa_supplicant/wpa_supplicant.conf
```

```
network={
    ssid="your wifi account"
    psk="your wifi password"
}
```

- Test WIFI

```bash
ping www.google.com
```


### SSH
- Enable SSH on Raspberry pi

> https://www.raspberrypi.org/documentation/remote-access/ssh/


- Get the IP of your raspberry pi 


```bash
hostname -I
```

- Test SSH connection from your remote machine


```bash
ssh pi@<IP>
```

- Set the SSH login without password


> http://www.linuxproblem.org/art_9.html


### Time elapse shots
- Prepare a BASH script for camera-shot

> https://www.raspberrypi.org/documentation/usage/camera/raspicam/raspistill.md

```bash
#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M%S")
raspistill -vf -hf -o /home/pi/camera/$DATE.jpg
```

- Set up a CRON job

> www.raspberrypi.org/documentation/usage/camera/raspicam/timelapse.md

```bash
crontab -e
```

```
* * * * * /home/pi/camera.sh 2>&1
```



### Upload images from raspberry pi to Dropbox
- Install the Dropbox Uploader
> https://github.com/andreafabrizi/Dropbox-Uploader/#usage


- Set up a CRON job for the dropbox transfer

```bash
crontab -e
```

```
# Upload images every 30 min
30 * * * * ./Dropbox-Uploader/dropbox_uploader.sh -f /home/pi/.dropbox_uploader upload /home/pi/camera/*.jpg  /

# Empty the camera folder every hour
* 1 * * * rm /home/pi/camera/*.jpg
```

### Stitch images to video
> https://nicholasnadeau.me/post/2020/5/converting-gopro-timelapse-to-a-video-or-gif-with-imagemagick-and-ffmpeg

- Rename Files to Sequential Numbers

```bash
ls | cat -n | while read n f; do mv "$f" "$n.jpg"; done
```

- Convert to a video

```
ffmpeg -framerate 30 -i %d.JPG -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
```


### Summary

<div class="imgcap">
<img src="/assets/time_elapse/final.png" height="300">
<div class="thecap">In testing.</div>
</div>