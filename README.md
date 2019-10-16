# Tripod :star:

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
Execution time took 0.7646613121032715 seconds
Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
0.7309989929199219 0.7803457379341125 0.3567299246788025 0.35346105694770813

Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
-0.5788195133209229 -0.7110175490379333 0.7020860314369202 0.8736361265182495

Science (from the Latin word scientia, meaning "knowledge")[1] is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
-0.27698490023612976 -0.503406822681427 0.9999860525131226 0.8280588984489441

Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
-0.439662367105484 -0.4755406975746155 -0.19494909048080444 -0.06704480201005936

Their contributions to mathematics, astronomy, and medicine entered and shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to provide explanations of events in the physical world based on natural causes
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
-0.20391632616519928 -0.3391631841659546 0.3518427908420563 0.65208500623703

Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country
The city proper is administratively known as the "City"), and has the same administrative level as that of a national county, being further subdivided into six sectors, each governed by a local mayor.
0.6992307305335999 0.7146259546279907 0.7043308615684509 0.49322831630706787
```

The four values represent: overall cosine distance, summary-based cosine distance, GST-based cosine distance and memory-based cosine distance. 


## Training you own models

This chapter is still under development

### Licensing

This project is licensed under the Apache V2 License. See [LICENSE](LICENSE) for more information.
