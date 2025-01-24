# fraud-exterminator-9000
<b>A <s>cutting-edge</s> simple application designed to detect counterfeit banknotes. Made for scientific purposes only. Nah.</b>
<br/>

<h3>Dataset</h3>
Dataset used for this project can be found here: https://www.kaggle.com/datasets/vivekgediya/banknote-authenticationcsv
<br/>
<b>Shoutout</b> to all indian YouTube guys who helped me to finish this work, no seriously. 

<h2>How to run and use this</h2>
<h3>Installation</h3>

Install all libraries specified in requirements.txt

<h3>How to use</h3>
To launch this you need to open cmd, and then:

<h3>Arguments</h3>
--stage, -s: Program stage. Can be <i>'train'</i> or <i>'predict'</i>.
<br/>
--image_folder, -im: Name of the folder where images for PREDICTION are stored. This folder <b>MUST</b> be placed inside <b>predict</b> folder. Use with <i>'predict' stage only</i>.
<br/>
--file, -f: Name of the CSV file with training and test data. Data files <b>MUST</b> be placed inside <b>data</b> folder. Use with <i>'train'</i> stage only.
<br/>
--model, -m: Model for training or prediction. There are 4 models available: <i>'logreg'</i>, <i>'r_forest'</i>, <i>'logreg_custom'</i>, <i>'svm_custom'</i>.
<br/>
--iterations, -i: Number of iterations used for model training.
<br/>
--learning_rate, -lr: Learning rate used for model training.
<br/>

<h3>Training</h3>

Example commands:
```console
python main.py -s 'train' -f 'BankNote_Authentication_01.csv' -m 'logreg' -i 1000 -lr 0.001
```
Learning rate and iterations can be skipped. Default arguments will be used then.
```console
python main.py -s 'train' -f 'BankNote_Authentication_01.csv' -m 'svm_custom' 
```
When training is finished, models are saved to <b>saved_params</b> folder. Also, metrics, confusion matrix, and ROC-AUC curve are presented.

<h3>Prediction</h3>
The dataset entries consist of 4 image metrics: variance, skewness, kurtosis, entropy and a class label.
To use the model for prediction on actual images, we need to extract these metrics. For these purposes, grayscaling and further metric extraction with scipy are used.

Example commands:
```console
python main.py -s 'predict' -m 'logreg_custom' -im 'test'
```
Program scans the provided folder for .jpg and .png images. Then, the model decides if the bills on these images are either legitimate or counterfeit.
Also, image metrics for every image are presented.
<br/>
I hope you find this manual to be helpful. 
<br/>
<img src='meme.jpg'/>