A simple model implemented with tensorflow for voiceprint

wav-->fbank-->lstm-->embedding-->train and test/softmax(pretrain)-->train 

dataset£ºhttps://datashare.is.ed.ac.uk/download/DS_10283_2651.zip

Hyperparameters used in the model

- duration of wavs used for training: 2000ms
- how long each frame of spectrograme: 25ms
- how far to move in time between two frames: 10ms
- numbers of coefficients of fbank: 40
- numbers of enrollment utts for each speaker: 5
- numbers of units for each layer of lstm: 128
- dimension of projection layer of lstm: 64
- number of layers of multi-lstm: 3
- dimension of linear layer on top of lstm: 64
- learning rate: 0.0001
- dropout prob: 0.5
- batch size: 80
- Each batch contains N = 8 speakers and M = 10 utterances per speaker.
- 
<strike>Each batch contains N = 64 speakers and M = 10 utterances per speaker.</strike>
