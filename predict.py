import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    # FIXME
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ${Transformer-Encoder}-------------------')
    print('Github: ${TranDucChinh}')
    print('Email: ${chinhtran2004@gmail.com}')
    print('---------------------------------------------------------------------')
    print('Training ${Transformer Encoder} model with hyper-params:')
    print('===========================')

    # FIXME
    # Do Training

