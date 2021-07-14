docker run -e DISPLAY=$DISPLAY \
           -v $HOME/.Xauthority:/root/.Xauthority:rw \
           -v /mnt/interns/qin/projects/defgrid-release:/work/defgrid \
           -v /mnt/interns/qin/data:/work/data \
           --ipc=host \
           --net host --rm -it defgrid bash
