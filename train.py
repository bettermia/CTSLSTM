
import json
from CTSLSTM import *

if __name__ == "__main__":
    # load hyperparameters
    hps = json.load(open('./hparam_files/HyperParameters.json', 'r'))

    # path for log saving
    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # model construction
    model = CTSLSTM(hps)

    # train and test
    model.run()