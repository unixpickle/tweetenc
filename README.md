# tweetenc

The goal of this project is to derive the latent variables behind tweets. To do this, I will be experimenting with various forms of sequence-to-sequence auto-encoders.

# Data

You can find some tweet data [here](http://help.sentiment140.com/for-students/). It was intended to be used for sentiment analysis, but it can be repurposed for this. However, it is slightly biased (only tweets with emoticons were used).

# Results

I trained a model with 768 LSTM cells per layer and a bottleneck layer with 1024 neurons. After a day of training on a Titan X, the model gets down to a cost of about 0.7 nats. The model is fairly good at reconstructions:

<table>
  <tr>
    <th>Original</th>
    <th>Reconstructed</th>
  </tr>
  <tr>
    <td>I hate my job.</td>
    <td>I hate my job.</td>
  </tr>
  <tr>
    <td>today will be a good day</td>
    <td>today will be a good day</td>
  </tr>
  <tr>
    <td>Well, that's my musical day set then.</td>
    <td>Well, that's my musical days then sleep.</td>
  </tr>
  <tr>
    <td>@unixpickle I am not sure if you're serious...</td>
    <td>@inupciline I am so tired your superfure...  ok.</td>
  </tr>
</table>

You can also use the model to interpolate between two tweets. Due to [this paper](https://arxiv.org/pdf/1511.06349v4.pdf), I suspect that I would get better interpolations if I used a variational auto-encoder (something I am looking into). For now, here's what we got:

<table>
  <tr><td>0.000</td><td>I hate my job.</td></tr>
  <tr><td>0.167</td><td>I hate my 10  one.</td></tr>
  <tr><td>0.333</td><td>I have my toddlering.</td></tr>
  <tr><td>0.500</td><td>I have my folding  trashes</td></tr>
  <tr><td>0.667</td><td>I love my friends at hang </td></tr>
  <tr><td>0.833</td><td>I love my friends and family</td></tr>
  <tr><td>1.000</td><td>I love my friends and family</td></tr>
</table>

<table>
  <tr><td>0.000</td><td>I hate my job.</td></tr>
  <tr><td>0.167</td><td>I hate my job.</td></tr>
  <tr><td>0.333</td><td>I had to be my agile.</td></tr>
  <tr><td>0.500</td><td>Ita do we had my big lonely</td></tr>
  <tr><td>0.667</td><td>today we blit a good hand</td></tr>
  <tr><td>0.833</td><td>today will be a good day</td></tr>
  <tr><td>1.000</td><td>today will be a good day</td></tr>
</table>
