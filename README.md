# zoidberg

run install_requirements, .sh if on linux or .bat if on windows

create a top level folder called chest_Xray where you unzip your dataset
```
├───chest_Xray
│   ├───test
│   │   ├───NORMAL
│   │   └───PNEUMONIA
│   ├───train
│   │   ├───NORMAL
│   │   └───PNEUMONIA
│   └───val
│       ├───NORMAL
│       └───PNEUMONIA
```


create a models folder at the same level as chest_Xray, 
you will have to manually place the models there after they are saved (for the moment)

change these values in train.py to match your need
```
  img_height = 256
  img_width = 256
  num_channels = 3  # Set to 1 for grayscale images
  batch_size = 32
  num_epochs = 5
``` 
After training is done, run inference.py, which will output a textfile with the results

inference.py will run inference on all images in the test folder for each model inside /model and output a detailed_results text file in the root folder
