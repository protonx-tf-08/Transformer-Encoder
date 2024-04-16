# TODO 3
import os
from argparse import ArgumentParser
from models import TransformerClassifier

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
  

    home_dir = os.getcwd()
    args = parser.parse_args()

    num_encoder_layers = args.num_encoder_layers
    d_model = args.d_model
 
    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${account}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')


    # Process data

    # Instantiate the model

    # Compile the model

    # Train the model

    # Evaluate the model
   