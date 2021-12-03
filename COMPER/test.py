import tensorflow as tf
import sys
from random import randrange
import click
from comper.agents.test_agent import TestAgent
from comper.config import parameters as param
import getpass
#export DISPLAY="`grep nameserver /etc/resolv.conf | sed 's/nameserver //'`:0"
#export PYTHONPATH=${PYTHONPATH}:.

#EXEMPLE 1 - python COMPER/main_test.py --rom freeway --display 1  --netparamsdir './netparams/exp4-freeway-1' --logdir './log/test/exp4-freeway-1'  --test_agent 1  
#EXEMPLE 2 - python COMPER/main_test.py --rom freeway  --netparamsdir './netparams/exp4-freeway-1' --logdir './log/test/exp4-freeway-1'  --test_agent 2 --salience_map 1 --salience_map_dir './salience_map_dir/exp4-freeway-1'
#EXEMPLE 3 - COMPER/main_test.py --display 1 --rom freeway  --netparamsdir './netparams/exp4-freeway-1' --logdir './log/test/exp4-freeway-1'  --test_agent 2 --pca_frames_clustering 1 --pca_frames_dir './pca_frames_dir/exp4-freeway-1'


@click.command()
@click.option("--rom", type=str, default="freeway",show_default=True)
@click.option("--screenon", is_flag=True,show_default=True)
@click.option("--logdir", type=str, default='dev',show_default=True)
@click.option("--netparamsdir", type=str, default='dev')
@click.option('--framesmode', type=click.Choice(['single', 'staked'],case_sensitive=False),default='single',show_default=True)

def main(rom,screenon,logdir,netparamsdir,framesmode):
    
    rom_file = str.encode("/home/"+getpass.getuser()+"/workspace/COMPER/COMPER/rom/"+rom+".bin")
    
    agent = TestAgent(rom_name =rom,
                    rom_file_path=rom_file,                    
                    log_dir=logdir,
                    nets_param_dir=netparamsdir,
                    display_screen=screenon,
                    framesmode = framesmode)
    agent.RunTest(1)
    
    

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print(" ")
