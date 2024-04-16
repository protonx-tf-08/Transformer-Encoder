# TODO 3
import os
from argparse import ArgumentParser
from models import TransformerClassifier
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--max-length", default=200,type=int)
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-heads", default=2, type=int)
    parser.add_argument("--num-encoder-layers",default = 2, type=int)
    parser.add_argument("--learning-rate",default=0.001,type=float)
    parser.add_argument("--dropout-rate", default=0.1, type = float)
  
    home_dir = os.getcwd()
    args = parser.parse_args()

    num_encoder_layers = args.num_encoder_layers
    d_model = args.d_model
 
    # Project Description

    print('---------------------Welcome to ${Transformer-Encoder}-------------------')
    print('Github: ${TranDucChinh}')
    print('Email: ${chinhtran2004@gmail.com}')
    print('---------------------------------------------------------------------')
    print('Training ${Transformer Encoder} model with hyper-params:')
    print('===========================')


    # Process data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.vocab_size)
    x_train = pad_sequences(x_train, maxlen = args.max_length)
    x_test = pad_sequences(x_test, maxlen= args.max_length)

    # Instantiate the model
    model = TransformerClassifier(
        num_encoder_layers=args.num_encoder_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=args.vocab_size,
        maximum_position_encoding=args.max_length
    )

    # Compile the model
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

    # Train the model
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.num_epochs, validation_data=(x_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=0)])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

   