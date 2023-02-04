# Notes
This module is to create a custom Tensorflow dataset that will take the following inputs:
- video file
- event timestamp
- frame step
- sequence length

It will then generate all possible sequences of frames surrounding the event timestamp.
Each sequence with have the specified sequence length.
Each sequence will be sampled from the original video frames at every "frame step".
Each sequence will have label 0 for every frames before the event and label 1 for every frames after the event. The label 1 signifies the event has occurred.