#!/bin/bash

hpc_loc="/home1/yulunzha/icaros/hearthstone/EvoStone/TestBed/DeckSearch_FCNN"
sftp yulunzha@discovery.usc.edu << EOF
  put -r bin/ ${hpc_loc}
  put -r config/experiment ${hpc_loc}/config
  put -r slurm/ ${hpc_loc}
  put -r resources/ ${hpc_loc}
  exit
EOF