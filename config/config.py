import argparse
import json

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UVIN')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--visible_device', type=str, default='3')
    parser.add_argument('--test_final_game', type=int, default=100)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--config_file', type=str)

    config = vars(parser.parse_args())

    cf = json.load(open(config['config_file']))
    config.update(cf)

    return config

if __name__=='__main__':
    print(get_config())