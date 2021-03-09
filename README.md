# EmotionDetection

Facial Expression Detection based on FER2013 Dataset (https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013)

Built into a script that load the model to predict emotions from frames captured from user's webcam. Training and testing notebook is provided as well. 

Model (model.json and weights.h5) provided is built using CNN and is able to achieve 71.86% on test data. 

(Emotions available: "Angry", "Disgusted", "Worried", "Happy", "Sad", "Terrified")


<h2> To run the emotiondetect.py </h2>

```
py emotiondetect.py --json [your model] --weights [your weights] 
```
For e.g. 
```
py emotiondetect.py --json model.json --weights weights.h5
```
