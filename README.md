# Playing Cards Classifier App (trained using PyTorch)

# Important Note
* Pre-trained model used was EfficientNet_B0 (model pre-trained on ImageNet dataset) : https://github.com/lukemelas/EfficientNet-PyTorch
* Custom model was trained using the smallest EfficientNet, model imported using torch image model (timm) : https://huggingface.co/timm
* Model Trained on 53 classes including Joker, refer to classes.txt for the list of classes
* For the purpose of using Streamlit cloud, the smallest model is chosen for a faster inference speed, with less accuracy
* Accuracy of the model is **97.36%**

# Link to app
* [https://dogclassifier-efficientnet.streamlit.app/](https://playingcardclassifier-efficientnet.streamlit.app/)https://playingcardclassifier-efficientnet.streamlit.app/






