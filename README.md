# Texture Recognition
This is my master thesis project. 

The main goal is to recognize different textures on the images from drones. 

###Current workflow:
- Apply filter bank to the whole image.
- Perform clustering of filter response vectors.
- Use filter response vectors distribution among clusters to teach neural network.
- Take another image, apply filter bank, find cluster centers.
- Provide NN with the new cluster centers to determine connection with previous clusters.

Project is still under progress.

### Used approaches:
- [Tiny-CNN](https://github.com/nyanp/tiny-cnn)
- [Filter Banks](http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)
- [Research paper by Jitenda Malik](http://www.cs.berkeley.edu/~malik/papers/MalikBLS.pdf)
