
Implemented CLR from this paper: https://arxiv.org/pdf/1506.01186.pdf

This did not change the accuracy in a positive way.  It actually reduced the accuracy.  I am unsure if it's actually doing what I though its doing.  

Regardless, I tried increasing the hidden layer output size.  This increased the model accuracy, breaking the 95% regime consistently.  The hidden layer had a count of 100 and pushing it past this number didn't change the accuracy in a meaningful way.


