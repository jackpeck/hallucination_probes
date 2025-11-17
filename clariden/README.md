## some config files


there are some config files related to the clariden cluster, e.g. [Dockerfile](./Dockerfile) of the image, should be some sbatch scripts etc.


The base `.sqsh` image with `uv` and repo cloned is saved at `/capstor/scratch/cscs/tkwiecinski/hallucination-probes/base.sqsh`


Notice, that there is no uv installed in the container (it seems that uv doesn't really like working with torch and global env).