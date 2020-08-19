# dnn_reco
IceCube DNN reconstruction

## Documentation
   Documentation can be found here: https://icecube.wisc.edu/~mhuennefeld/docs/dnn_reco/html/pages/about.html

## Installation Instructions
    # clone repository and call this from dnn_reco directory:
    pip install -e ./ --process-dependency-links

## Example Usage

    # Create data transformation model:
    # We must first create a transformation model that will take care of data normalization and transformation
    python create_trafo_model.py /PATH/TO/MY/YAML/CONFIG/FILE
    
    # Train model:
    # This step can be run with as many config files and settings as you wish.
    # The settings and number of training iterations is automatically logged and will be exported together
    # with the final model.
    python train_model.py /PATH/TO/MY/YAML/CONFIG/FILE
    
    # Export model:
    # Once the model is trained, we can export it, so that it can be used to reconstruct IceCube events with the provided I3Module
    python export_model.py /PATH/TO/MY/YAML/CONFIG/FILE -s /PATH/TO/CONFIG/FILE/USED/TO/CREATE/TRAINING/DATA -o OUTPUT/Directory

    # More documentation can be found here: https://icecube.wisc.edu/~mhuennefeld/docs/dnn_reco/html/
   

