FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3


#SHELL ["/bin/bash", "--login", "-CMD"]

ARG USERNAME=daniel
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN sudo apt-get --assume-yes install libosmesa6-dev
RUN sudo apt-get --assume-yes install libc-dev
RUN sudo pip install patchelf
RUN sudo apt-get --assume-yer install htop
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
WORKDIR /home/$USERNAME/COMPER-GYM
COPY . .
RUN sudo chown -R $USERNAME /home/$USERNAME/COMPER-GYM
RUN sudo mv mujoco /home/$USERNAME/.mujoco
#CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USERNAME/.mujoco/mjpro150/bin
#CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/daniel/$USERNAME/mujoco210/bin
RUN pip install faiss-gpu
RUN pip install gym
#RUN pip install -U 'mujoco-py<1.50.2,>=1.50.1'
RUN pip install -U 'mujoco-py<2.2,>=2.1'
#RUN pip install mujoco
RUN pip install gym[mujoco]
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USERNAME/.mujoco/mujoco210/bin" >> /home/$USERNAME/.bashrc

