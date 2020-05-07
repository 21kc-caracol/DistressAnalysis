"""

    from tqdm import tqdm

    import evaluate_screamClf



    def different_data_size():
        screamGlobals= evaluate_screamClf.global_For_Clf()

        for size in tqdm([150,200,250,300,350,400]):
            screamGlobals.try_lower_amount = size
            evaluate_screamClf.experiment_data_size(screamGlobals)

    def different_model():
        #  screamGlobals= evaluate_screamClf.global_For_Clf()

        for size in tqdm([50,100,150,200,250,300,350,400]):
            screamGlobals.try_lower_amount = size
            #  evaluate_screamClf.experiment_data_size(screamGlobals)
            evaluate_screamClf.experiment_data_size()

"""
import json
from pathlib import Path
from keras.models import model_from_json

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


if __name__ == '__main__':
        # screamGlobals = evaluate_screamClf.global_For_Clf()
        # different_data_size()
        # different_model()
    #  model reconstruction from JSON:
    path = Path('models')
    current_model_path = path / "scream_model.json"
    with open(current_model_path) as file:
        model_data = json.load(file)
    model = model_from_json(model_data)
    model.summary()
    print(len(model.layers))
    #  print(model.get_layer(index=1))
    config = model.get_config()
    print(config['name'])


    #  pretty(config, indent=0)
    """
    for key, val in config.items():
        print
        key, "=>", val
    """
    print("layers: ")
    for layer_num in range(0,len(model.layers)):
        print(model.get_layer(index=layer_num))
