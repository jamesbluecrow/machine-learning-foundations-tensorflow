1. Import import tensorflow
2. Load the data
3. Reshape data to include filter parameter / color depth
4. Normalise data to be between 0-1 (divide by 255)
5. Prepare model and declare neurons: Sequential, Conv2D, MaxPooling2D, Flatten, Dense...
6. Declare callbacks to receive calls during training
7. Train the neural network: model.compile & model.fit
8. Test: model.evaluate