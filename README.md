# Unearth.Demo.MLCodes
Find codes in a string in ML.NET - are char-grams a good idea?

Often you will get text data with embedded codes, not always in the same position, and have to interpret these codes from a string.

This demo creates 2 models using the same training data (airline flight info strings), one uses Text Featurization and the other uses character tri-grams (looking for IATA Aircraft type codes like '73H').

With a training set of 10K samples the typical result is: Text Featurizing Model: 93% Accurate, CharGram Model: 98.5% accurate, but training with all the extra features takes much (10x) longer. Still the char-gram approach is worthwhile in this case. 

HOWEVER, if you train with 'ManyFlightCodes' (100K samples) this becomes more like 97.8% -> 98.5%.

Adding Text Featurization or extra character n-grams (say quad-grams) makes training slower and helps accuracy only imperceptably. 

The conclusion is: As always, 'It Depends'. If you have a limited set of training data then tri (or bi, or quad - depends on what you are looking for) -grams are a good idea. But the 'catch all' Text Featurization facility in ML.NET is very good, especially when paired with the SDCA algorithm and fed lots of training data. The LogisticRegression algorithm works best when you have lots of features (n-grams in this case).
