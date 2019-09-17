# Tripod

Tripod is a tool/ML model for computing latent representations for large sequences. It has been used on source code and text and it has applications such as:
* Malicious code detection
* Sentiment analysis
* Information/code indexing and retrieval
* Anomaly Detection/ Unsupervised Learning

**For a quick demo** you can try the [Colaboratory Notebook with GPU acceleration](https://colab.research.google.com/drive/1_Qb8-KgIlpXwNtN5TdmBikEelN7djWaL). If you want to modify the text, open the Notebook in playground mode.

Link to white paper will be available soon.


**Note:** With minor modifications, you can apply the same model to images and audio

## Quick start guide

The fastest way to get started with Tripod is to use the precompiled PIP package. Make sure you have python3 and pip installed and use the following command:

```bash
$ pip3 install tripod-ml
```

Requirements:
* python3 
* installed dependencies (requirements.txt)
## Usage

Tripod comes with an easy to use API. After downloading the pip package:


```python
from tripod.api import Tripod
tripod=Tripod()
tripod.load('wiki-103')
examples=['Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.', 'Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes', 'Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country', 'The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.']

from scipy import spatial

for ii in range(len(examples)):
     for jj in range(ii+1, len(examples)):
         s=1 - spatial.distance.cosine(rez[ii], rez[jj])
         s_sum=1 - spatial.distance.cosine(rez[ii][0:300], rez[jj][0:300])
         s_gst=1 - spatial.distance.cosine(rez[ii][300:600], rez[jj][300:600])
         s_mem=1 - spatial.distance.cosine(rez[ii][600:900], rez[jj][600:900])
         print (examples[ii]+'\n\n'+examples[jj])
         print ('{0} {1} {2} {3}'.format(s, s_sum, s_gst, s_mem))
         print('\n\n\n')
```

The output should look like this:

```text
Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
0.9926257133483887 0.9913334846496582 0.9999998807907104 0.9999991655349731

Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
-0.8557108044624329 -0.9669785499572754 -0.305904358625412 -0.1784113049507141

Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
0.9188222289085388 0.9868599772453308 0.5696542263031006 0.04632262885570526

Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
-0.8592780828475952 -0.9673091173171997 -0.30608630180358887 -0.17940381169319153

Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
0.9194520711898804 0.9849761128425598 0.5695375800132751 0.0449962355196476

Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
-0.7460803985595703 -0.958350658416748 0.36185234785079956 0.5165910124778748
```

## Training you own models

This chapter is still under development

The four values represent: overall cosine distance, summary-based cosine distance, GST-based cosine distance and memory-based cosine distance. 
### Licensing

This project is licensed under the Apache V2 License. See [LICENSE](LICENSE) for more information.